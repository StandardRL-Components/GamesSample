import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


# Set the SDL video driver to "dummy" for headless operation
os.environ["SDL_VIDEODRIVER"] = "dummy"

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use ↑ and ↓ to rotate your paddle. Deflect the ball to the top of the screen to advance through the tunnel."
    )

    game_description = (
        "Navigate a twisting isometric tunnel by deflecting a bouncing ball with a rotating paddle to reach the end."
    )

    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    PADDLE_Y_RATIO = 0.85
    PADDLE_WIDTH = 80
    PADDLE_THICKNESS = 8
    PADDLE_ROTATION_SPEED = 0.08
    BALL_RADIUS = 8
    TUNNEL_WIDTH = 200
    NUM_LEVELS = 5
    SEGMENTS_PER_LEVEL = 20
    SEGMENT_LENGTH = 150
    
    # Colors
    COLOR_BG = (10, 15, 25)
    COLOR_BALL = (255, 255, 0)
    COLOR_PADDLE = (0, 200, 255)
    COLOR_PADDLE_OUTLINE = (100, 255, 255)
    COLOR_TEXT = (220, 220, 220)
    
    # Level colors for tunnel
    LEVEL_COLORS = [
        (40, 40, 80),   # Level 1
        (60, 40, 80),   # Level 2
        (80, 40, 60),   # Level 3
        (80, 40, 40),   # Level 4
        (100, 30, 30),  # Level 5
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 72)
        
        self.render_mode = render_mode
        self.np_random = None

        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.lives = 0
        self.current_level = 0
        self.progress = 0.0
        self.ball_pos = pygame.math.Vector2(0, 0)
        self.ball_vel = pygame.math.Vector2(0, 0)
        self.paddle_angle = 0.0
        self.tunnel_path = []
        self.particles = []
        self.level_transition_timer = 0
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.lives = 3
        self.current_level = 1
        self.progress = 0.0
        self.paddle_angle = 0.0
        self.particles = []
        self.level_transition_timer = 90  # 3 seconds at 30fps

        self._generate_tunnel()
        self._reset_ball()

        return self._get_observation(), self._get_info()

    def _reset_ball(self):
        cam_pos = self._get_camera_pos()
        paddle_screen_pos = pygame.math.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT * self.PADDLE_Y_RATIO)
        paddle_world_pos = self._screen_to_world(paddle_screen_pos, cam_pos)
        
        self.ball_pos = paddle_world_pos + pygame.math.Vector2(0, -50)
        speed = self._get_ball_speed()
        self.ball_vel = pygame.math.Vector2(self.np_random.uniform(-0.5, 0.5), -1).normalize() * speed

    def _get_ball_speed(self):
        return 5.0 + (self.current_level - 1) * 0.75

    def step(self, action):
        movement, _, _ = action
        reward = 0.0
        terminated = False

        if self.level_transition_timer > 0:
            self.level_transition_timer -= 1
        else:
            # 1. Handle Input
            if movement == 1:  # Up -> Rotate Clockwise
                self.paddle_angle += self.PADDLE_ROTATION_SPEED
            elif movement == 2:  # Down -> Rotate Counter-Clockwise
                self.paddle_angle -= self.PADDLE_ROTATION_SPEED
            self.paddle_angle %= (2 * math.pi)

            # 2. Update Ball
            self.ball_pos += self.ball_vel
            
            # 3. Handle Collisions and Boundaries
            reward += self._handle_collisions()

            # 4. Check for level up or game over
            old_level = self.current_level
            self.current_level = min(self.NUM_LEVELS, 1 + int(self.progress / (self.SEGMENTS_PER_LEVEL * self.SEGMENT_LENGTH)))

            if self.current_level > old_level:
                reward += 10.0
                self.level_transition_timer = 90
                if self.current_level > self.NUM_LEVELS:
                    reward += 100.0
                    self.game_over = True
        
        self._update_particles()
        
        self.steps += 1
        self.score += reward
        
        if self.lives <= 0:
            self.game_over = True

        terminated = self.game_over or self.steps >= 5000
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_collisions(self):
        reward = 0
        cam_pos = self._get_camera_pos()

        # Ball vs Paddle
        paddle_center_screen = pygame.math.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT * self.PADDLE_Y_RATIO)
        paddle_center_world = self._screen_to_world(paddle_center_screen, cam_pos)
        
        angle_vec = pygame.math.Vector2(math.cos(self.paddle_angle), math.sin(self.paddle_angle))
        p1 = paddle_center_world - angle_vec * self.PADDLE_WIDTH / 2
        p2 = paddle_center_world + angle_vec * self.PADDLE_WIDTH / 2

        # Simple line-circle collision
        line_vec = p2 - p1
        if line_vec.length() > 0:
            t = max(0, min(1, (self.ball_pos - p1).dot(line_vec) / line_vec.length_squared()))
            closest_point = p1 + t * line_vec
            dist_to_paddle = self.ball_pos.distance_to(closest_point)

            if dist_to_paddle < self.BALL_RADIUS:
                reward += 0.1
                self._create_particles(self.ball_pos, self.COLOR_PADDLE, 15)
                
                # Reflect velocity
                paddle_normal = pygame.math.Vector2(-math.sin(self.paddle_angle), math.cos(self.paddle_angle))
                self.ball_vel = self.ball_vel.reflect(paddle_normal)
                self.ball_vel.scale_to_length(self._get_ball_speed())
                
                # Push ball out of paddle
                self.ball_pos = closest_point + paddle_normal * self.BALL_RADIUS


        # Ball vs Walls
        path_idx, closest_point_on_path = self._find_closest_segment_and_point(self.ball_pos)
        if closest_point_on_path:
            dist_from_center = self.ball_pos.distance_to(closest_point_on_path)

            if dist_from_center > self.TUNNEL_WIDTH / 2:
                self._create_particles(self.ball_pos, self.LEVEL_COLORS[self.current_level - 1], 10)
                
                # Reflect velocity based on wall normal
                wall_normal = (self.ball_pos - closest_point_on_path).normalize()
                self.ball_vel = self.ball_vel.reflect(wall_normal)
                
                # Push ball back inside tunnel
                self.ball_pos = closest_point_on_path + wall_normal * (self.TUNNEL_WIDTH / 2 - 1)
            
            # Danger zone reward
            if dist_from_center > self.TUNNEL_WIDTH / 2 * 0.8:
                reward -= 0.02
            
        # Ball vs Screen Boundaries
        ball_screen_pos = self._world_to_screen(self.ball_pos, cam_pos)

        if ball_screen_pos.y > self.SCREEN_HEIGHT + self.BALL_RADIUS:
            self.lives -= 1
            reward -= 5.0
            if self.lives > 0:
                self._reset_ball()
        
        if ball_screen_pos.y < -self.BALL_RADIUS:
            self.progress += self.ball_vel.length() * 1.5 # Scroll faster than ball speed
            self._reset_ball()

        return reward

    def _get_observation(self):
        # --- Background ---
        self.screen.fill(self.COLOR_BG)
        
        # --- Game Elements ---
        cam_pos = self._get_camera_pos()
        self._render_tunnel(cam_pos)
        self._render_particles(cam_pos)
        self._render_paddle(cam_pos)
        self._render_ball(cam_pos)
        
        # --- UI ---
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "level": self.current_level,
            "progress": self.progress,
        }

    # --- Generation and State Helpers ---

    def _generate_tunnel(self):
        self.tunnel_path = [pygame.math.Vector2(0, 0)]
        angle = -math.pi / 2  # Start pointing straight "up" in world space
        
        total_segments = self.NUM_LEVELS * self.SEGMENTS_PER_LEVEL
        for i in range(total_segments):
            level = 1 + int(i / self.SEGMENTS_PER_LEVEL)
            max_angle_change = math.radians(15 * level)
            angle_change = self.np_random.uniform(-max_angle_change, max_angle_change)
            angle += angle_change
            
            # Prevent turning back on itself too sharply
            if angle < -math.pi: angle = -math.pi
            if angle > 0: angle = 0

            new_point = self.tunnel_path[-1] + pygame.math.Vector2(
                math.cos(angle), math.sin(angle)
            ) * self.SEGMENT_LENGTH
            self.tunnel_path.append(new_point)

    def _get_camera_pos(self):
        # Camera follows the progress marker along the path
        path_dist = 0
        for i in range(len(self.tunnel_path) - 1):
            p1, p2 = self.tunnel_path[i], self.tunnel_path[i+1]
            segment_len = p1.distance_to(p2)
            if path_dist + segment_len >= self.progress:
                ratio = (self.progress - path_dist) / segment_len if segment_len > 0 else 0
                return p1.lerp(p2, ratio)
            path_dist += segment_len
        return self.tunnel_path[-1]

    def _find_closest_segment_and_point(self, pos):
        min_dist_sq = float('inf')
        closest_point = None
        closest_idx = -1

        # Heuristic: check segments around the camera's position
        cam_idx, _ = self._find_closest_segment_and_point_on_path(self._get_camera_pos(), self.tunnel_path)
        start_idx = max(0, cam_idx - 10)
        end_idx = min(len(self.tunnel_path) - 1, cam_idx + 20)

        for i in range(start_idx, end_idx):
            p1, p2 = self.tunnel_path[i], self.tunnel_path[i+1]
            line_vec = p2 - p1
            if line_vec.length_squared() == 0: continue
            
            t = max(0, min(1, (pos - p1).dot(line_vec) / line_vec.length_squared()))
            point_on_segment = p1 + t * line_vec
            dist_sq = (pos - point_on_segment).length_squared()
            
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                closest_point = point_on_segment
                closest_idx = i
        
        return closest_idx, closest_point

    @staticmethod
    def _find_closest_segment_and_point_on_path(pos, path):
        min_dist_sq = float('inf')
        closest_point = None
        closest_idx = -1
        for i in range(len(path) - 1):
            p1, p2 = path[i], path[i+1]
            line_vec = p2 - p1
            if line_vec.length_squared() == 0: continue
            t = max(0, min(1, (pos - p1).dot(line_vec) / line_vec.length_squared()))
            point_on_segment = p1 + t * line_vec
            dist_sq = (pos - point_on_segment).length_squared()
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                closest_point = point_on_segment
                closest_idx = i
        return closest_idx, closest_point

    # --- Coordinate Transformations ---

    def _world_to_screen(self, pos, cam_pos):
        rel_pos = pos - cam_pos
        iso_x = rel_pos.x - rel_pos.y
        iso_y = (rel_pos.x + rel_pos.y) * 0.5
        return pygame.math.Vector2(
            self.SCREEN_WIDTH / 2 + iso_x,
            self.SCREEN_HEIGHT * 0.6 + iso_y # Camera view is shifted down
        )

    def _screen_to_world(self, screen_pos, cam_pos):
        iso_x = screen_pos.x - self.SCREEN_WIDTH / 2
        iso_y = screen_pos.y - self.SCREEN_HEIGHT * 0.6
        
        rel_y = (2 * iso_y - iso_x) / 2
        rel_x = iso_x + rel_y
        
        return pygame.math.Vector2(rel_x, rel_y) + cam_pos

    # --- Particle System ---

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            vel = pygame.math.Vector2(self.np_random.uniform(-1, 1), self.np_random.uniform(-1, 1)).normalize() * self.np_random.uniform(1, 3)
            self.particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "life": self.np_random.integers(15, 30),
                "max_life": 30,
                "color": color
            })

    def _update_particles(self):
        for p in self.particles:
            p["pos"] += p["vel"]
            p["vel"] *= 0.95 # friction
            p["life"] -= 1
        self.particles = [p for p in self.particles if p["life"] > 0]

    # --- Rendering ---

    def _render_tunnel(self, cam_pos):
        cam_idx, _ = self._find_closest_segment_and_point_on_path(cam_pos, self.tunnel_path)
        start_idx = max(0, cam_idx - 15)
        end_idx = min(len(self.tunnel_path) - 1, cam_idx + 25)

        tunnel_color = self.LEVEL_COLORS[self.current_level - 1]
        
        for i in range(start_idx, end_idx):
            p1, p2 = self.tunnel_path[i], self.tunnel_path[i+1]
            if p1.distance_to(p2) == 0: continue
            normal = (p2 - p1).normalize().rotate(90)
            
            # Wall lines
            wall_p1_l = self._world_to_screen(p1 + normal * self.TUNNEL_WIDTH / 2, cam_pos)
            wall_p2_l = self._world_to_screen(p2 + normal * self.TUNNEL_WIDTH / 2, cam_pos)
            wall_p1_r = self._world_to_screen(p1 - normal * self.TUNNEL_WIDTH / 2, cam_pos)
            wall_p2_r = self._world_to_screen(p2 - normal * self.TUNNEL_WIDTH / 2, cam_pos)
            
            pygame.draw.aaline(self.screen, tunnel_color, wall_p1_l, wall_p2_l)
            pygame.draw.aaline(self.screen, tunnel_color, wall_p1_r, wall_p2_r)

            # Floor grid lines
            floor_p1 = self._world_to_screen(p1, cam_pos)
            floor_p2 = self._world_to_screen(p2, cam_pos)
            grid_color = (tunnel_color[0]//2, tunnel_color[1]//2, tunnel_color[2]//2)
            pygame.draw.aaline(self.screen, grid_color, floor_p1, floor_p2)
            pygame.draw.aaline(self.screen, grid_color, wall_p1_l, wall_p1_r)

    def _render_ball(self, cam_pos):
        screen_pos = self._world_to_screen(self.ball_pos, cam_pos)
        
        # Glow effect
        glow_radius = int(self.BALL_RADIUS * 1.8)
        glow_color = (self.COLOR_BALL[0]//2, self.COLOR_BALL[1]//2, self.COLOR_BALL[2]//2)
        pygame.gfxdraw.filled_circle(self.screen, int(screen_pos.x), int(screen_pos.y), glow_radius, glow_color)
        pygame.gfxdraw.aacircle(self.screen, int(screen_pos.x), int(screen_pos.y), glow_radius, glow_color)
        
        pygame.gfxdraw.filled_circle(self.screen, int(screen_pos.x), int(screen_pos.y), self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, int(screen_pos.x), int(screen_pos.y), self.BALL_RADIUS, self.COLOR_BALL)

    def _render_paddle(self, cam_pos):
        center_screen = pygame.math.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT * self.PADDLE_Y_RATIO)
        angle_vec = pygame.math.Vector2(1, 0).rotate_rad(self.paddle_angle)
        
        p1 = center_screen - angle_vec * self.PADDLE_WIDTH / 2
        p2 = center_screen + angle_vec * self.PADDLE_WIDTH / 2
        
        pygame.draw.line(self.screen, self.COLOR_PADDLE, p1, p2, self.PADDLE_THICKNESS)
        pygame.draw.aaline(self.screen, self.COLOR_PADDLE_OUTLINE, p1, p2)

    def _render_particles(self, cam_pos):
        for p in self.particles:
            alpha = int(255 * (p["life"] / p["max_life"]))
            color = (*p["color"], alpha)
            radius = int(self.BALL_RADIUS * 0.5 * (p["life"] / p["max_life"]))
            if radius > 0:
                screen_pos = self._world_to_screen(p["pos"], cam_pos)
                pygame.gfxdraw.filled_circle(self.screen, int(screen_pos.x), int(screen_pos.y), radius, color)

    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Lives
        lives_text = self.font_small.render(f"LIVES: {self.lives}", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (self.SCREEN_WIDTH - lives_text.get_width() - 10, 10))

        # Level transition text
        if self.level_transition_timer > 0:
            alpha = 255
            if self.level_transition_timer < 30: # Fade out
                alpha = int(255 * (self.level_transition_timer / 30))
            
            if self.current_level > self.NUM_LEVELS:
                 level_str = "YOU WIN!"
            else:
                 level_str = f"LEVEL {self.current_level}"

            level_text_surf = self.font_large.render(level_str, True, self.COLOR_TEXT)
            level_text_surf.set_alpha(alpha)
            text_rect = level_text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(level_text_surf, text_rect)

    def close(self):
        pygame.quit()

if __name__ == "__main__":
    # This block allows you to play the game directly
    # Make sure to set the video driver to something that works on your system
    # For linux, 'x11' is common. For Windows, you might not need to set it.
    # For headless servers, use 'dummy'.
    try:
        os.environ["SDL_VIDEODRIVER"] = "x11"
        pygame.display.init()
        pygame.font.init() # Font init is also needed for display mode
    except pygame.error:
        print("Failed to set SDL_VIDEODRIVER to 'x11', using 'dummy' mode.")
        os.environ["SDL_VIDEODRIVER"] = "dummy"


    env = GameEnv(render_mode="rgb_array")
    
    obs, info = env.reset(seed=42)
    terminated = False
    
    # Pygame setup for human play
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Tunnel Runner")
    clock = pygame.time.Clock()

    action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)

    print(GameEnv.user_guide)

    while not terminated:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        
        # Reset action
        action.fill(0)
        
        # Map keys to actions
        if keys[pygame.K_UP]:
            action[0] = 1 # Rotate CW
        elif keys[pygame.K_DOWN]:
            action[0] = 2 # Rotate CCW

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Game Over! Final Score: {int(info['score'])}")
            # Wait a bit before closing
            pygame.time.wait(2000)
            terminated = True

        clock.tick(30) # 30 FPS

    env.close()