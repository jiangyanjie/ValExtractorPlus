package ast;

import lombok.Getter;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;
import org.eclipse.jdt.core.dom.ASTParser;
import org.eclipse.jdt.core.dom.CompilationUnit;
import org.w3c.dom.Document;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;
import utils.Utils;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashSet;

@Slf4j
public class ProjectsParser {
    @Getter
    @Setter
    protected HashSet<String> targetJavaFiles;
    protected ArrayList<String> allJavaFiles;
    protected String[] sourcetreeEntries;
    protected String[] encodings;
    protected String[] classpathEntries;

    private Path[] targetPaths;
    private ArrayList<String> classpathEntriesList;
    private Path repoPath;

    public ProjectsParser(Path[] targetPaths, Path projectPath, Path repoPath) throws Exception {
        this.targetPaths = targetPaths;
        this.repoPath = repoPath;
        targetJavaFiles = new HashSet<>();
        allJavaFiles = new ArrayList<>();
        classpathEntriesList = new ArrayList<>();
        traverseFile(projectPath.toFile());
        classpathEntries = classpathEntriesList.toArray(new String[classpathEntriesList.size()]);
        parseSourceEntries();
    }

    private void traverseFile(File root) {
        if (root.isFile()) {
            if (root.getName().endsWith(".java")) {
//                log.info("Parsing file: " + root.getAbsolutePath());
                allJavaFiles.add(root.getAbsolutePath());
                for (Path targetPath : targetPaths)
                    if (root.getAbsolutePath().startsWith(targetPath.toString())) {
                        targetJavaFiles.add(root.getAbsolutePath());
                        break;
                    }
            }
//            if (root.getName().equals("pom.xml")) {
//                classpathEntriesList.addAll(parsePOM(root.getAbsolutePath()));
//            }
            return;
        } else if (root.isDirectory()) {
            for (File f : root.listFiles())
                traverseFile(f);
        }
    }


    private void parseSourceEntries() throws Exception {
        HashSet<String> sourceRootSet = new HashSet<String>();
        for (String javaFile : allJavaFiles) {
            ASTParser astParser = Utils.getNewASTParser();
            String code = Utils.getCodeFromFile(new File(javaFile));
            astParser.setSource(code.toCharArray());
            try{
                CompilationUnit compilationUnit = (CompilationUnit) astParser.createAST(null);
                if (compilationUnit.getPackage() == null)
                    continue;
                String rootPath = parseRootPath(javaFile, compilationUnit.getPackage().getName().toString());
                if (!rootPath.equals("")) sourceRootSet.add(rootPath);
            }catch (Exception e){
//                log.error("Error in parsing file: " + javaFile);
                LightASTParser lightASTParser =new LightASTParser(code.toCharArray());
                CompilationUnit cu = lightASTParser.getCompilationUnit();
                if (cu==null || cu.getPackage() == null)
                    continue;
                String rootPath = parseRootPath(javaFile,cu.getPackage().getName().toString());
                if (!rootPath.equals("")) sourceRootSet.add(rootPath);
            }

        }
        sourcetreeEntries = new String[sourceRootSet.size()];
        encodings = new String[sourceRootSet.size()];
        int index = 0;
        for (String sourceRoot : sourceRootSet) {
            sourcetreeEntries[index] = sourceRoot;
            encodings[index] = "utf-8";
            index++;
        }
    }

    private ArrayList<String> parsePOM(String pomPath) {
        ArrayList<String> tempClasspathEntriesList = new ArrayList<String>();
        DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
        try {
            DocumentBuilder builder = factory.newDocumentBuilder();
            Document document = builder.parse(pomPath);
            NodeList dependencies = document.getElementsByTagName("dependency");
            for (int i = 0; i < dependencies.getLength(); i++) {
                Node dependency = dependencies.item(i);
                String groupId = "";
                String artifactId = "";
                String version = "";
                NodeList childNodes = dependency.getChildNodes();
                for (int j = 0; j < childNodes.getLength(); j++) {
                    Node childNode = childNodes.item(j);
                    if (childNode.getNodeName().equals("groupId"))
                        groupId = childNode.getTextContent();
                    else if (childNode.getNodeName().equals("artifactId"))
                        artifactId = childNode.getTextContent();
                    else if (childNode.getNodeName().equals("version"))
                        version = childNode.getTextContent();
                }
                if (groupId.equals("") || artifactId.equals("") || version.equals("")) continue;
                groupId = groupId.replaceAll("\\.", "\\\\");
                Path classEntry = repoPath.resolve(groupId);
                classEntry = classEntry.resolve(artifactId);
                classEntry = classEntry.resolve(version);
                //String classEntry = repoPath + groupId + "\\" + artifactId + "\\" + version;
                File classEntryFolder = classEntry.toFile();
                if (classEntryFolder.exists()) tempClasspathEntriesList.add(classEntry.toString());
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        return tempClasspathEntriesList;
    }

    private String parseRootPath(String filePath, String packageName) {
        String t = packageName.replaceAll("\\.", "\\\\");
        Path relativePath = Paths.get(t);
        Path absolutePath = Paths.get(filePath).resolveSibling("");
        int end = absolutePath.toString().lastIndexOf(relativePath.toString());
        if (end == -1) return "";
        return absolutePath.toString().substring(0, end);
    }

    public CompilationUnit parse(String path) throws IOException {
        String code = Utils.getCodeFromFile(new File(path));
//        System.out.println(code);
        ASTParser astParser = Utils.getNewASTParser(sourcetreeEntries, encodings);
        astParser.setSource(code.toCharArray());

        try {
            CompilationUnit cu = (CompilationUnit) astParser.createAST(null);
            // 设置preserveWhiteSpace属性为true，保留空白和空行
//            cu.setProperty(DefaultCommentMapper.USE_INTERNAL_JAVADOC_PARSER_PROPERTY, Boolean.TRUE);

//            cu.recordModifications();
//            System.out.println(cu.getLineNumber(cu.getLength()-1)+ ", " +code.split("\\r?\\n|\\r").length)  ;
            return cu;
        }catch (Exception e){
            LightASTParser lightASTParser =new LightASTParser(code.toCharArray());
            CompilationUnit cu = lightASTParser.getCompilationUnit();
            cu.recordModifications();
            return cu;
        }
    }

    public String[] getSourcetreeEntries() {
        return sourcetreeEntries;
    }

    public String[] getEncodings() {
        return encodings;
    }

    public String[] getClasspathEntries() {
        return classpathEntries;
    }
}
