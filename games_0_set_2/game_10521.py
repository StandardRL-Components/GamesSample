import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T10:32:26.869403
# Source Brief: brief_00521.md
# Brief Index: 521
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import collections

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # --- User-facing metadata ---
    game_description = (
        "Juggle bouncing balls with a paddle and guide them into the target zone before time runs out."
    )
    user_guide = (
        "Controls: Use the ← and → arrow keys to move the platform left and right."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 60
    MAX_EPISODE_STEPS = 90 * FPS  # 90 seconds

    # Colors (Catppuccin Mocha)
    COLOR_BG = (30, 30, 46)        # Base
    COLOR_WALL = (69, 71, 90)      # Surface1
    COLOR_PLATFORM = (205, 214, 244) # Text
    COLOR_TARGET = (166, 227, 161)   # Green
    COLOR_TEXT = (205, 214, 244)     # Text
    BALL_COLORS = [
        (243, 139, 168), # Red
        (137, 180, 250), # Blue
        (249, 226, 175), # Yellow
        (180, 190, 254), # Mauve (sub for Green to contrast with target)
    ]
    
    # Game Physics
    PLATFORM_SPEED = 5
    PLATFORM_WIDTH = 100
    PLATFORM_HEIGHT = 10
    BALL_RADIUS = 10
    MAX_BALL_SPEED = 15
    PLATFORM_KICK_MULTIPLIER = 0.3
    PLATFORM_PENALTY_FRAMES = 10
    PLATFORM_PENALTY_MULTIPLIER = 0.8

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_game_over = pygame.font.SysFont("Consolas", 64, bold=True)

        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.timer = 0
        self.score = 0
        self.game_over = False
        self.platform = None
        self.balls = []
        self.last_ball_distances = []
        self.platform_speed_penalty_timer = 0
        
        # --- Maze and Target Definition ---
        self._define_maze_and_target()

        # --- Final setup ---
        # self.reset() is called by the environment wrapper
        
    def _define_maze_and_target(self):
        self.target_pos = pygame.Vector2(self.SCREEN_WIDTH * 0.15, self.SCREEN_HEIGHT * 0.8)
        self.target_radius = 40

        self.walls = [
            # Boundaries
            pygame.Rect(0, 0, self.SCREEN_WIDTH, 10),
            pygame.Rect(0, self.SCREEN_HEIGHT - 10, self.SCREEN_WIDTH, 10),
            pygame.Rect(0, 0, 10, self.SCREEN_HEIGHT),
            pygame.Rect(self.SCREEN_WIDTH - 10, 0, 10, self.SCREEN_HEIGHT),
            # Internal obstacles
            pygame.Rect(150, 100, 20, 150),
            pygame.Rect(self.SCREEN_WIDTH - 170, 100, 20, 150),
            pygame.Rect(250, 200, 140, 20),
        ]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.timer = self.MAX_EPISODE_STEPS
        self.score = 0
        self.game_over = False
        self.platform_speed_penalty_timer = 0

        self.platform = pygame.Rect(
            (self.SCREEN_WIDTH - self.PLATFORM_WIDTH) / 2,
            self.SCREEN_HEIGHT - 50,
            self.PLATFORM_WIDTH,
            self.PLATFORM_HEIGHT
        )

        self.balls = []
        for i in range(4):
            while True:
                pos = pygame.Vector2(
                    self.np_random.uniform(self.BALL_RADIUS + 10, self.SCREEN_WIDTH - self.BALL_RADIUS - 10),
                    self.np_random.uniform(self.BALL_RADIUS + 10, self.SCREEN_HEIGHT / 2)
                )
                # Ensure balls don't spawn inside a wall
                ball_rect = pygame.Rect(pos.x - self.BALL_RADIUS, pos.y - self.BALL_RADIUS, self.BALL_RADIUS*2, self.BALL_RADIUS*2)
                if ball_rect.collidelist(self.walls) == -1:
                    break

            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(2, 6)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            
            self.balls.append({
                "pos": pos,
                "vel": vel,
                "color": self.BALL_COLORS[i],
                "captured": False,
                "trail": collections.deque(maxlen=15)
            })

        self.last_ball_distances = [
            (ball["pos"] - self.target_pos).length() for ball in self.balls
        ]

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.timer -= 1
        
        # --- 1. Handle Actions & Update Platform ---
        movement = action[0]
        platform_move_intent = 0
        if movement == 3:  # Left
            platform_move_intent = -1
        elif movement == 4: # Right
            platform_move_intent = 1

        current_platform_speed = self.PLATFORM_SPEED
        if self.platform_speed_penalty_timer > 0:
            current_platform_speed *= self.PLATFORM_PENALTY_MULTIPLIER
            self.platform_speed_penalty_timer -= 1

        self.platform.x += platform_move_intent * current_platform_speed
        
        # Platform-wall collision
        collided_wall = self.platform.collidelist(self.walls)
        if collided_wall != -1:
            wall = self.walls[collided_wall]
            if platform_move_intent > 0: # Moving right
                self.platform.right = wall.left
            elif platform_move_intent < 0: # Moving left
                self.platform.left = wall.right
            self.platform_speed_penalty_timer = self.PLATFORM_PENALTY_FRAMES
        
        # Clamp platform to screen
        self.platform.left = max(10, self.platform.left)
        self.platform.right = min(self.SCREEN_WIDTH - 10, self.platform.right)

        # --- 2. Update Balls ---
        step_reward = 0
        for i, ball in enumerate(self.balls):
            if ball["captured"]:
                continue

            ball["trail"].append(ball["pos"].copy())
            ball["pos"] += ball["vel"]

            # Ball-wall collision
            ball_rect = pygame.Rect(ball["pos"].x - self.BALL_RADIUS, ball["pos"].y - self.BALL_RADIUS, self.BALL_RADIUS*2, self.BALL_RADIUS*2)
            collided_wall_idx = ball_rect.collidelist(self.walls)
            if collided_wall_idx != -1:
                wall = self.walls[collided_wall_idx]
                # Simple but effective collision response
                overlap = ball_rect.clip(wall)
                if overlap.width > overlap.height: # Horizontal collision
                    ball["vel"].y *= -1
                    if ball["pos"].y > wall.centery:
                        ball["pos"].y = wall.bottom + self.BALL_RADIUS
                    else:
                        ball["pos"].y = wall.top - self.BALL_RADIUS
                else: # Vertical collision
                    ball["vel"].x *= -1
                    if ball["pos"].x > wall.centerx:
                        ball["pos"].x = wall.right + self.BALL_RADIUS
                    else:
                        ball["pos"].x = wall.left - self.BALL_RADIUS

            # Ball-platform collision
            if ball_rect.colliderect(self.platform) and ball["vel"].y > 0:
                ball["pos"].y = self.platform.top - self.BALL_RADIUS
                ball["vel"].y *= -1
                
                # Impart platform momentum
                platform_imparted_vel = platform_move_intent * current_platform_speed * self.PLATFORM_KICK_MULTIPLIER
                ball["vel"].x += platform_imparted_vel
                
                # Clamp speed
                speed = ball["vel"].length()
                if speed > self.MAX_BALL_SPEED:
                    ball["vel"] = ball["vel"].normalize() * self.MAX_BALL_SPEED
                
            # Ball-target check
            dist_to_target = (ball["pos"] - self.target_pos).length()
            if dist_to_target < self.target_radius:
                ball["captured"] = True
                self.score += 1
                step_reward += 1.0
            else:
                # Continuous reward for getting closer
                if dist_to_target < self.last_ball_distances[i]:
                    step_reward += 0.01
                else:
                    step_reward -= 0.01
                self.last_ball_distances[i] = dist_to_target

        # --- 3. Check Termination ---
        terminated = False
        truncated = False
        
        all_captured = all(b["captured"] for b in self.balls)
        if all_captured:
            terminated = True
            step_reward += 100.0  # Victory bonus
            self.game_over = True
        elif self.timer <= 0:
            truncated = True # Use truncated for time limit
            step_reward -= 100.0  # Failure penalty
            self.game_over = True
        
        return self._get_observation(), step_reward, terminated, truncated, self._get_info()

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)

        # Render maze walls
        for wall in self.walls:
            pygame.draw.rect(self.screen, self.COLOR_WALL, wall)

        # Render target area (glow effect)
        for i in range(self.target_radius, 0, -2):
            alpha = 60 * (1 - i / self.target_radius)
            pygame.gfxdraw.filled_circle(
                self.screen, int(self.target_pos.x), int(self.target_pos.y), i,
                (*self.COLOR_TARGET, int(alpha))
            )
        pygame.gfxdraw.aacircle(
            self.screen, int(self.target_pos.x), int(self.target_pos.y), self.target_radius,
            self.COLOR_TARGET
        )

        # Render balls, trails, and speed indicators
        for ball in self.balls:
            # Trail
            if len(ball["trail"]) > 1:
                for i, pos in enumerate(ball["trail"]):
                    alpha = int(100 * (i / len(ball["trail"])))
                    color = (*ball["color"], alpha)
                    pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), self.BALL_RADIUS // 2, color)
            
            # Ball
            if not ball["captured"]:
                pygame.gfxdraw.filled_circle(self.screen, int(ball["pos"].x), int(ball["pos"].y), self.BALL_RADIUS, ball["color"])
                pygame.gfxdraw.aacircle(self.screen, int(ball["pos"].x), int(ball["pos"].y), self.BALL_RADIUS, (0,0,0,50))
                # Speed indicator
                speed_indicator_radius = int(max(0, min(self.BALL_RADIUS-2, ball["vel"].length())))
                if speed_indicator_radius > 0:
                     pygame.gfxdraw.filled_circle(self.screen, int(ball["pos"].x), int(ball["pos"].y), speed_indicator_radius, (*self.COLOR_BG, 150))

        # Render platform
        platform_color = self.COLOR_PLATFORM
        if self.platform_speed_penalty_timer > 0:
            # Flash red when penalized
            platform_color = (243, 139, 168)
        pygame.draw.rect(self.screen, platform_color, self.platform, border_radius=3)
        pygame.draw.rect(self.screen, self.COLOR_BG, self.platform.inflate(-4, -4), border_radius=3)

        # Render UI
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_ui(self):
        # Timer display
        timer_text = f"TIME: {self.timer // self.FPS:02d}"
        timer_surf = self.font_ui.render(timer_text, True, self.COLOR_TEXT)
        self.screen.blit(timer_surf, (self.SCREEN_WIDTH - timer_surf.get_width() - 20, 20))

        # Score display (captured balls)
        score_text = f"CAPTURED: {self.score}/4"
        score_surf = self.font_ui.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (20, 20))

        # Game over message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            if self.score == 4:
                msg_text = "YOU WIN!"
                msg_color = self.COLOR_TARGET
            else:
                msg_text = "TIME'S UP!"
                msg_color = self.BALL_COLORS[0] # Red
            
            msg_surf = self.font_game_over.render(msg_text, True, msg_color)
            msg_rect = msg_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(msg_surf, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "timer": self.timer,
        }
        
    def close(self):
        pygame.quit()

# Example usage:
if __name__ == '__main__':
    # Un-comment the line below to run with a display
    os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv()
    
    # --- Manual Play Loop ---
    obs, info = env.reset()
    done = False
    
    # Use a separate display for human play
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Ball Juggler")
    
    running = True
    while running:
        # Default action: no-op (movement=0)
        # The other action components [1] and [2] are not used in this game.
        action = [0, 0, 0] 
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            action[0] = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            action[0] = 4
        
        if done:
            # Allow restarting after a delay
            pygame.time.wait(2000)
            obs, info = env.reset()
            done = False

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Convert observation back to a surface for display
        # The observation is (H, W, C), but pygame needs (W, H, C) surfarray
        # Transpose from (H, W, C) to (W, H, C)
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(env.FPS)

    env.close()