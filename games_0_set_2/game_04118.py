
# Generated: 2025-08-28T01:29:07.837867
# Source Brief: brief_04118.md
# Brief Index: 4118

        
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
        "Controls: ←→ to aim, ↑↓ to adjust power. Press space to putt. Hold shift to reset your aim."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A minimalist isometric golf game. Aim your shot, control the power, and try to sink the ball in three strokes or less."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30 # Not used for advance, but for physics timing if needed

        # Colors
        self.COLOR_BG = (25, 28, 36)
        self.COLOR_GREEN = (65, 152, 10)
        self.COLOR_GREEN_DARK = (55, 130, 8)
        self.COLOR_BALL = (240, 240, 240)
        self.COLOR_SHADOW = (0, 0, 0, 50)
        self.COLOR_HOLE = (10, 10, 10)
        self.COLOR_FLAG_POLE = (200, 200, 200)
        self.COLOR_FLAG = (220, 50, 50)
        self.COLOR_AIM_LINE = (50, 150, 255, 150)
        self.COLOR_POWER_BAR_BG = (50, 50, 50)
        self.COLOR_POWER_BAR_FILL = (255, 200, 0)
        self.COLOR_TEXT = (230, 230, 230)
        self.COLOR_TEXT_SHADOW = (20, 20, 20)

        # Game parameters
        self.MAX_STROKES = 5
        self.MAX_STEPS = 1000
        self.WIN_STROKE_TARGET = 3
        self.GREEN_RADIUS = 160
        self.HOLE_RADIUS = 8
        self.BALL_RADIUS = 5
        self.FRICTION = 0.985
        self.MIN_POWER = 0.1
        self.MAX_POWER = 1.0
        self.POWER_INCREMENT = 0.05
        self.ANGLE_INCREMENT = math.radians(2.5)
        self.MAX_PUTT_FORCE = 15.0

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)
        
        # --- State Variables ---
        self.ball_pos = None
        self.ball_vel = None
        self.hole_pos = None
        self.stroke_count = None
        self.aim_angle = None
        self.putt_power = None
        self.is_ball_moving = None
        self.last_dist_to_hole = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.game_over_message = ""
        
        # Initialize state variables
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset game state
        self.ball_pos = np.array([0.0, -self.GREEN_RADIUS * 0.8])
        self.ball_vel = np.array([0.0, 0.0])
        
        # Place hole randomly in the upper half of the green
        angle = self.np_random.uniform(math.pi / 4, 3 * math.pi / 4)
        radius = self.np_random.uniform(0.2, 0.8) * self.GREEN_RADIUS
        self.hole_pos = np.array([math.cos(angle) * radius, math.sin(angle) * radius])

        self.stroke_count = 0
        self.aim_angle = math.pi / 2 # Aim straight up initially
        self.putt_power = self.MIN_POWER
        self.is_ball_moving = False
        self.last_dist_to_hole = np.linalg.norm(self.ball_pos - self.hole_pos)
        
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.game_over_message = ""
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.steps += 1
        reward = -0.01  # Small penalty for each step to encourage efficiency
        terminated = False

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_pressed = action[1] == 1  # Boolean
        shift_pressed = action[2] == 1  # Boolean

        if self.is_ball_moving:
            self._update_ball_physics()
            
            current_dist = np.linalg.norm(self.ball_pos - self.hole_pos)
            dist_reward = (self.last_dist_to_hole - current_dist)
            reward += dist_reward
            self.last_dist_to_hole = current_dist

            if np.linalg.norm(self.ball_vel) < 0.05:
                self.is_ball_moving = False
                self.ball_vel = np.array([0.0, 0.0])
                # // sound: ball_stop.wav
                
                # Check for win/loss conditions only when ball stops
                if current_dist < self.HOLE_RADIUS:
                    # // sound: win.wav
                    reward += 50
                    if self.stroke_count <= self.WIN_STROKE_TARGET:
                        reward += 50 # Bonus for skilled play
                    self.game_over_message = f"SUNK IN {self.stroke_count}!"
                    terminated = True
                elif self.stroke_count >= self.MAX_STROKES:
                    # // sound: lose.wav
                    reward -= 5
                    self.game_over_message = "OUT OF STROKES"
                    terminated = True

        else: # Aiming phase
            if movement == 1: # Up
                self.putt_power = min(self.MAX_POWER, self.putt_power + self.POWER_INCREMENT)
            elif movement == 2: # Down
                self.putt_power = max(self.MIN_POWER, self.putt_power - self.POWER_INCREMENT)
            elif movement == 3: # Left
                self.aim_angle -= self.ANGLE_INCREMENT
            elif movement == 4: # Right
                self.aim_angle += self.ANGLE_INCREMENT
            
            if shift_pressed:
                self.aim_angle = math.atan2(self.hole_pos[1] - self.ball_pos[1], self.hole_pos[0] - self.ball_pos[0])
                self.putt_power = self.MIN_POWER

            if space_pressed:
                # // sound: putt.wav
                self.stroke_count += 1
                force = self.MIN_POWER + (self.putt_power * (self.MAX_POWER - self.MIN_POWER)) * self.MAX_PUTT_FORCE
                self.ball_vel = np.array([math.cos(self.aim_angle), math.sin(self.aim_angle)]) * force
                self.is_ball_moving = True
                self.putt_power = self.MIN_POWER # Reset power after putt
        
        # Check for out of bounds
        if np.linalg.norm(self.ball_pos) > self.GREEN_RADIUS + self.BALL_RADIUS:
            # // sound: lose_oob.wav
            reward -= 10
            self.game_over_message = "OUT OF BOUNDS"
            terminated = True
        
        if self.steps >= self.MAX_STEPS:
            terminated = True
        
        self.game_over = terminated
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _update_ball_physics(self):
        self.ball_pos += self.ball_vel
        self.ball_vel *= self.FRICTION

    def _to_iso(self, x, y):
        iso_x = self.WIDTH / 2 + (x - y)
        iso_y = self.HEIGHT / 2 * 0.8 + (x + y) / 2
        return int(iso_x), int(iso_y)

    def _render_game(self):
        # Green
        pygame.gfxdraw.filled_ellipse(self.screen, self.WIDTH // 2, self.HEIGHT // 2, int(self.GREEN_RADIUS * 1.05), int(self.GREEN_RADIUS * 0.55), self.COLOR_GREEN_DARK)
        pygame.gfxdraw.filled_ellipse(self.screen, self.WIDTH // 2, self.HEIGHT // 2, self.GREEN_RADIUS, self.GREEN_RADIUS // 2, self.COLOR_GREEN)

        # Hole
        hole_iso = self._to_iso(self.hole_pos[0], self.hole_pos[1])
        pygame.gfxdraw.filled_circle(self.screen, hole_iso[0], hole_iso[1], self.HOLE_RADIUS, self.COLOR_HOLE)
        
        # Flag
        pygame.draw.line(self.screen, self.COLOR_FLAG_POLE, (hole_iso[0], hole_iso[1]), (hole_iso[0], hole_iso[1] - 30), 2)
        pygame.gfxdraw.filled_trigon(self.screen, hole_iso[0], hole_iso[1] - 30, hole_iso[0], hole_iso[1] - 20, hole_iso[0] + 15, hole_iso[1] - 25, self.COLOR_FLAG)
        
        # Aiming guide
        if not self.is_ball_moving:
            self._render_aim_guide()

        # Ball Shadow
        ball_iso = self._to_iso(self.ball_pos[0], self.ball_pos[1])
        shadow_rect = pygame.Rect(0, 0, self.BALL_RADIUS * 2.5, self.BALL_RADIUS * 1.5)
        shadow_rect.center = (ball_iso[0], ball_iso[1] + self.BALL_RADIUS)
        shadow_surf = pygame.Surface(shadow_rect.size, pygame.SRCALPHA)
        pygame.draw.ellipse(shadow_surf, self.COLOR_SHADOW, (0, 0, *shadow_rect.size))
        self.screen.blit(shadow_surf, shadow_rect.topleft)

        # Ball
        pygame.gfxdraw.filled_circle(self.screen, ball_iso[0], ball_iso[1], self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, ball_iso[0], ball_iso[1], self.BALL_RADIUS, (0,0,0,50))

    def _render_aim_guide(self):
        # Trajectory prediction
        sim_pos = self.ball_pos.copy()
        force = self.MIN_POWER + (self.putt_power * (self.MAX_POWER - self.MIN_POWER)) * self.MAX_PUTT_FORCE
        sim_vel = np.array([math.cos(self.aim_angle), math.sin(self.aim_angle)]) * force

        for i in range(60): # Simulate 60 physics steps
            sim_pos += sim_vel
            sim_vel *= self.FRICTION
            if i % 4 == 0:
                dot_iso = self._to_iso(sim_pos[0], sim_pos[1])
                # Fade out dots over time
                alpha = max(0, 200 - i * 3)
                pygame.gfxdraw.filled_circle(self.screen, dot_iso[0], dot_iso[1], 2, (*self.COLOR_AIM_LINE[:3], alpha))

    def _render_ui(self):
        # --- Text Helper ---
        def draw_text(text, font, color, pos, shadow_color=None):
            if shadow_color:
                text_surf_shadow = font.render(text, True, shadow_color)
                self.screen.blit(text_surf_shadow, (pos[0] + 2, pos[1] + 2))
            text_surf = font.render(text, True, color)
            self.screen.blit(text_surf, pos)

        # Stroke Count
        stroke_text = f"STROKE: {self.stroke_count} / {self.MAX_STROKES}"
        draw_text(stroke_text, self.font_small, self.COLOR_TEXT, (10, 10), self.COLOR_TEXT_SHADOW)

        # Power Bar
        bar_width, bar_height = 150, 20
        bar_x, bar_y = self.WIDTH - bar_width - 10, 10
        
        # Background
        pygame.draw.rect(self.screen, self.COLOR_POWER_BAR_BG, (bar_x, bar_y, bar_width, bar_height), border_radius=4)
        
        # Fill
        fill_width = (self.putt_power / self.MAX_POWER) * bar_width
        pygame.draw.rect(self.screen, self.COLOR_POWER_BAR_FILL, (bar_x, bar_y, fill_width, bar_height), border_radius=4)
        
        # Border
        pygame.draw.rect(self.screen, self.COLOR_TEXT, (bar_x, bar_y, bar_width, bar_height), 2, border_radius=4)
        draw_text("POWER", self.font_small, self.COLOR_TEXT, (bar_x - 60, 8), self.COLOR_TEXT_SHADOW)

        # Game Over Message
        if self.game_over:
            s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            s.fill((0,0,0,128))
            self.screen.blit(s, (0,0))
            
            text_surf = self.font_large.render(self.game_over_message, True, self.COLOR_TEXT)
            text_rect = text_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(text_surf, text_rect)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stroke_count": self.stroke_count,
            "ball_pos": self.ball_pos,
            "dist_to_hole": np.linalg.norm(self.ball_pos - self.hole_pos)
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        print("Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment for manual play
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    env.validate_implementation()
    
    # --- Manual Play Loop ---
    obs, info = env.reset()
    done = False
    
    # Pygame setup for display
    pygame.display.set_caption("Isometric Golf")
    display_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    action = env.action_space.sample()
    action = [0, 0, 0] # Start with no-op
    
    running = True
    while running:
        # Map keyboard inputs to actions for manual play
        mov, space, shift = 0, 0, 0
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: mov = 1
        elif keys[pygame.K_DOWN]: mov = 2
        elif keys[pygame.K_LEFT]: mov = 3
        elif keys[pygame.K_RIGHT]: mov = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
            
        action = [mov, space, shift]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: # Reset on 'r' key
                    obs, info = env.reset()
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Strokes: {info['stroke_count']}")
            # Render final frame with message
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            display_screen.blit(surf, (0, 0))
            pygame.display.flip()
            # Wait for a moment before auto-resetting
            pygame.time.wait(2000)
            obs, info = env.reset()
        
        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(60) # Control the speed of manual play
        
    env.close()