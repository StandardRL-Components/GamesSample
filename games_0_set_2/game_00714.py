
# Generated: 2025-08-27T14:32:38.006387
# Source Brief: brief_00714.md
# Brief Index: 714

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use ↑ and ↓ to move your paddle vertically. Avoid letting the ball hit the tunnel walls."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate a glowing ball through a twisting neon tunnel in Tunnel Pong, a fast-paced arcade game where precision and timing are key."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and world dimensions
        self.W, self.H = 640, 400
        self.FPS = 30
        self.TUNNEL_LENGTH = 10000 # pixels
        self.MAX_STEPS = 2000 # Increased to allow for longer gameplay

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.H, self.W, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.W, self.H))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 36)
        self.font_msg = pygame.font.Font(None, 72)
        
        # Colors
        self.COLOR_BG = (10, 5, 25)
        self.COLOR_PADDLE = (0, 255, 255) # Cyan
        self.COLOR_BALL = (255, 255, 255) # White
        self.COLOR_TUNNEL_MAIN = (120, 0, 255) # Purple
        self.COLOR_TUNNEL_GLOW = (50, 0, 100) # Darker Purple
        self.COLOR_SPARK = (255, 180, 0) # Orange
        self.COLOR_TEXT = (255, 255, 255)
        
        # Initialize state variables
        self.ball_pos = pygame.math.Vector2(0, 0)
        self.ball_vel = pygame.math.Vector2(0, 0)
        self.paddle_y = 0
        self.camera_x = 0
        self.particles = []
        self.tunnel_top_pts = []
        self.tunnel_bottom_pts = []
        self.current_segment_index = 0
        self.last_ball_x_dir = 1
        self.steps = 0
        self.score = 0
        self.time_remaining = 0
        self.game_over = False
        self.win = False
        self.np_random = None

        self.reset()

        # self.validate_implementation() # Optional validation call

    def _generate_tunnel(self):
        self.tunnel_top_pts = []
        self.tunnel_bottom_pts = []
        segment_len = 50.0
        num_segments = int(self.TUNNEL_LENGTH / segment_len)
        y_center = self.H / 2
        
        amp = 0
        freq = self.np_random.uniform(0.001, 0.002)
        y_offset = self.np_random.uniform(0, math.pi * 2)
        
        for i in range(num_segments + 1):
            x = i * segment_len
            
            # Start straight
            if i < 5:
                amp_multiplier = 0
            else:
                # Gradually increase curvature
                amp_multiplier = min(1.0, (i - 5) / (num_segments * 0.5))
            
            amp = amp_multiplier * self.H * 0.25

            y_center_offset = amp * math.sin(freq * x + y_offset)
            
            tunnel_width = 150 - 50 * amp_multiplier # Tunnel narrows as it gets curvier
            
            top_y = self.H / 2 + y_center_offset - tunnel_width / 2
            bottom_y = self.H / 2 + y_center_offset + tunnel_width / 2
            
            self.tunnel_top_pts.append((x, top_y))
            self.tunnel_bottom_pts.append((x, bottom_y))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.np_random = np.random.default_rng(seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.time_remaining = 60.0

        # Paddle
        self.paddle_y = self.H / 2
        self.PADDLE_HEIGHT = 80
        self.PADDLE_WIDTH = 15
        self.PADDLE_SPEED = 10
        self.PADDLE_SCREEN_X = 80

        # Ball
        self.BALL_RADIUS = 10
        self.BALL_SPEED = 12
        self.ball_pos = pygame.math.Vector2(self.PADDLE_SCREEN_X + 50, self.H / 2)
        initial_angle = self.np_random.uniform(-0.1, 0.1)
        self.ball_vel = pygame.math.Vector2(math.cos(initial_angle), math.sin(initial_angle)) * self.BALL_SPEED
        
        self.last_ball_x_dir = 1 if self.ball_vel.x > 0 else -1

        # Tunnel
        self._generate_tunnel()
        self.current_segment_index = 0

        self.particles = []
        self.camera_x = 0
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        terminated = False
        
        if not self.game_over:
            # Unpack factorized action
            movement = action[0]  # 0-4: none/up/down/left/right
            
            # --- UPDATE PADDLE ---
            if movement == 1: # Up
                self.paddle_y -= self.PADDLE_SPEED
            elif movement == 2: # Down
                self.paddle_y += self.PADDLE_SPEED
            
            self.paddle_y = np.clip(self.paddle_y, self.PADDLE_HEIGHT / 2, self.H - self.PADDLE_HEIGHT / 2)

            # --- UPDATE BALL ---
            self.ball_pos += self.ball_vel
            
            # --- REWARD for moving towards goal ---
            if self.ball_vel.x < 0:
                reward -= 0.01 # Penalty for moving backwards
            
            # --- PADDLE COLLISION ---
            paddle_rect = pygame.Rect(self.PADDLE_SCREEN_X - self.PADDLE_WIDTH/2, self.paddle_y - self.PADDLE_HEIGHT/2, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)
            ball_world_rect = pygame.Rect(self.ball_pos.x - self.BALL_RADIUS, self.ball_pos.y - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)
            
            # Convert paddle rect to world coordinates for collision check
            paddle_world_rect = paddle_rect.copy()
            paddle_world_rect.x += self.camera_x

            if paddle_world_rect.colliderect(ball_world_rect) and self.ball_vel.x < 0:
                # Sound: paddle_hit.wav
                reward += 0.1
                self.score += 1
                
                # Reflect velocity
                self.ball_vel.x *= -1
                
                # Add "spin" based on where it hit the paddle
                offset = (self.ball_pos.y - self.paddle_y) / (self.PADDLE_HEIGHT / 2)
                self.ball_vel.y += offset * 4.0
                
                # Normalize to maintain constant speed
                self.ball_vel.scale_to_length(self.BALL_SPEED)
                
                # Prevent sticking
                self.ball_pos.x = paddle_world_rect.right + self.BALL_RADIUS
                
                self._create_particles(self.ball_pos, 20, self.COLOR_SPARK)
            
            # --- TUNNEL COLLISION ---
            top_y = self._get_wall_y(self.ball_pos.x, self.tunnel_top_pts)
            bottom_y = self._get_wall_y(self.ball_pos.x, self.tunnel_bottom_pts)
            
            if self.ball_pos.y - self.BALL_RADIUS < top_y or self.ball_pos.y + self.BALL_RADIUS > bottom_y:
                # Sound: explosion.wav
                self.game_over = True
                terminated = True
                reward = -100
                self._create_particles(self.ball_pos, 50, self.COLOR_SPARK)
            
            # --- UPDATE SEGMENT PROGRESS ---
            new_segment_idx = int(self.ball_pos.x / 50.0)
            if new_segment_idx > self.current_segment_index:
                # Sound: segment_pass.wav
                self.current_segment_index = new_segment_idx
                reward += 1.0
                self.score += 10
            
            # --- UPDATE CAMERA ---
            self.camera_x = self.ball_pos.x - self.PADDLE_SCREEN_X - 50

            # --- UPDATE TIMER ---
            self.time_remaining -= 1.0 / self.FPS
        
        # --- UPDATE PARTICLES ---
        self._update_particles()
        
        # --- CHECK TERMINATION CONDITIONS ---
        if self.ball_pos.x >= self.TUNNEL_LENGTH:
            # Sound: win.wav
            self.game_over = True
            self.win = True
            terminated = True
            reward = 100
        elif self.time_remaining <= 0:
            self.game_over = True
            terminated = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True
        
        self.steps += 1
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _get_wall_y(self, x, points):
        if x < points[0][0]: return points[0][1]
        if x > points[-1][0]: return points[-1][1]
        
        # Find segment
        segment_idx = int(x / 50.0)
        if segment_idx + 1 >= len(points):
            return points[-1][1]
            
        p1 = points[segment_idx]
        p2 = points[segment_idx + 1]
        
        # Linear interpolation
        t = (x - p1[0]) / (p2[0] - p1[0]) if (p2[0] - p1[0]) != 0 else 0
        return p1[1] + t * (p2[1] - p1[1])

    def _create_particles(self, pos, count, color):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            vel = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "life": self.np_random.uniform(10, 25), # frames
                "color": color,
                "radius": self.np_random.uniform(1, 3)
            })

    def _update_particles(self):
        for p in self.particles:
            p["pos"] += p["vel"]
            p["vel"] *= 0.95 # friction
            p["life"] -= 1
        self.particles = [p for p in self.particles if p["life"] > 0]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # --- RENDER TUNNEL ---
        visible_points_top = []
        visible_points_bottom = []
        
        start_idx = max(0, int((self.camera_x - 100) / 50.0))
        end_idx = min(len(self.tunnel_top_pts), int((self.camera_x + self.W + 100) / 50.0) + 1)

        for i in range(start_idx, end_idx):
            # Top
            x, y = self.tunnel_top_pts[i]
            visible_points_top.append((x - self.camera_x, y))
            # Bottom
            x, y = self.tunnel_bottom_pts[i]
            visible_points_bottom.append((x - self.camera_x, y))

        if len(visible_points_top) > 1:
            for i in range(10, 0, -2): # Glow effect
                pygame.draw.aalines(self.screen, self.COLOR_TUNNEL_GLOW, False, [(x,y+i) for x,y in visible_points_top])
                pygame.draw.aalines(self.screen, self.COLOR_TUNNEL_GLOW, False, [(x,y-i) for x,y in visible_points_bottom])
            pygame.draw.aalines(self.screen, self.COLOR_TUNNEL_MAIN, False, visible_points_top, 2)
            pygame.draw.aalines(self.screen, self.COLOR_TUNNEL_MAIN, False, visible_points_bottom, 2)

        # --- RENDER PARTICLES ---
        for p in self.particles:
            screen_pos = (int(p["pos"].x - self.camera_x), int(p["pos"].y))
            alpha = int(255 * (p["life"] / 25.0))
            color = p["color"]
            pygame.gfxdraw.filled_circle(self.screen, screen_pos[0], screen_pos[1], int(p["radius"]), (*color, alpha))

        # --- RENDER PADDLE ---
        paddle_rect = pygame.Rect(self.PADDLE_SCREEN_X - self.PADDLE_WIDTH/2, self.paddle_y - self.PADDLE_HEIGHT/2, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)
        for i in range(5): # Glow
            glow_rect = paddle_rect.inflate(i*4, i*4)
            alpha = 100 - i * 20
            shape_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(shape_surf, (*self.COLOR_PADDLE, alpha), (0, 0, *glow_rect.size), border_radius=5)
            self.screen.blit(shape_surf, glow_rect.topleft)
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, paddle_rect, border_radius=5)

        # --- RENDER BALL ---
        ball_screen_pos = (int(self.ball_pos.x - self.camera_x), int(self.ball_pos.y))
        for i in range(5): # Glow
            alpha = 150 - i * 30
            pygame.gfxdraw.filled_circle(self.screen, ball_screen_pos[0], ball_screen_pos[1], self.BALL_RADIUS + i*2, (*self.COLOR_BALL, alpha))
        pygame.gfxdraw.filled_circle(self.screen, ball_screen_pos[0], ball_screen_pos[1], self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, ball_screen_pos[0], ball_screen_pos[1], self.BALL_RADIUS, self.COLOR_BALL)
        
    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.W - score_text.get_width() - 20, 10))
        
        # Time
        time_text = self.font_ui.render(f"TIME: {max(0, self.time_remaining):.1f}", True, self.COLOR_TEXT)
        self.screen.blit(time_text, (20, 10))
        
        # Game Over Message
        if self.game_over:
            msg = "YOU WIN!" if self.win else "GAME OVER"
            color = (0, 255, 0) if self.win else (255, 0, 0)
            msg_text = self.font_msg.render(msg, True, color)
            msg_rect = msg_text.get_rect(center=(self.W / 2, self.H / 2))
            self.screen.blit(msg_text, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining,
            "ball_pos": (self.ball_pos.x, self.ball_pos.y),
            "ball_vel": (self.ball_vel.x, self.ball_vel.y)
        }
        
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.H, self.W, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.H, self.W, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.H, self.W, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv()
    
    # --- Manual Play ---
    # To play manually, you need to install pygame and run this script.
    # The environment will be rendered to a pygame window.
    
    obs, info = env.reset()
    terminated = False
    
    # Replace the screen surface with a display surface for manual play
    env.screen = pygame.display.set_mode((env.W, env.H))
    pygame.display.set_caption("Tunnel Pong")
    
    action = env.action_space.sample()
    action.fill(0) # Start with no-op

    while not terminated:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        keys = pygame.key.get_pressed()
        
        # Reset action
        action.fill(0)
        
        # Map keys to actions
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # The observation is already rendered to env.screen,
        # so we just need to update the display
        pygame.display.flip()
        
        # Control the frame rate
        env.clock.tick(env.FPS)
        
    print(f"Game Over! Final Score: {info['score']}")
    env.close()