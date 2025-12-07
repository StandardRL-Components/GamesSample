
# Generated: 2025-08-28T01:57:57.359453
# Source Brief: brief_04292.md
# Brief Index: 4292

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to aim the launcher. Press space to fire a ball. "
        "Clear all blocks before time or balls run out."
    )

    # Must be a short,user-facing description of the game:
    game_description = (
        "A fast-paced, top-down block-breaking puzzle game. "
        "Strategically aim and launch balls to clear the screen and maximize your score."
    )

    # Frames auto-advance for time limits and smooth graphics.
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.GAME_DURATION_SECONDS = 60
        self.MAX_STEPS = self.GAME_DURATION_SECONDS * self.FPS # Cap steps at game duration
        self.INITIAL_BALLS = 15
        self.BALL_RADIUS = 7
        self.BALL_SPEED = 8
        self.LAUNCHER_LENGTH = 30
        
        # Colors
        self.COLOR_BG = (15, 20, 35)
        self.COLOR_GRID = (30, 40, 60)
        self.COLOR_PLAYER = (255, 255, 255)
        self.COLOR_UI_BG = (10, 15, 25, 180)
        self.COLOR_TEXT = (220, 220, 240)
        self.BLOCK_COLORS = {
            1: (0, 255, 128),   # Green
            2: (0, 128, 255),   # Blue
            3: (255, 64, 128)    # Red
        }

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)
        
        # Initialize state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = 0
        self.launcher_angle = 0.0
        self.launcher_pos = (0, 0)
        self.balls_remaining = 0
        self.active_balls = []
        self.particles = []
        self.blocks = []
        self.space_was_held = False
        self.win_condition = False
        self.loss_condition = False
        
        self.reset()
        
        # Run validation check
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_condition = False
        self.loss_condition = False
        self.time_remaining = self.GAME_DURATION_SECONDS * self.FPS
        
        self.launcher_angle = -90.0
        self.launcher_pos = (self.WIDTH // 2, self.HEIGHT - 20)
        
        self.balls_remaining = self.INITIAL_BALLS
        self.active_balls = []
        self.particles = []
        
        self.space_was_held = True # Prevent launch on first frame
        
        self._generate_blocks()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        reward = 0
        
        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        
        if not self.game_over:
            # Handle player input
            self._handle_input(movement, space_held)

            # Update game logic
            self.steps += 1
            self.time_remaining -= 1
            
            reward += self._update_balls()
            self._update_particles()
        
        # Check termination conditions
        terminated = self._check_termination()
        
        if self.win_condition:
            reward += 100
        elif self.loss_condition:
            reward -= 100

        self.game_over = terminated
            
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_held):
        # Rotate launcher
        if movement == 3:  # Left
            self.launcher_angle -= 2.0
        elif movement == 4:  # Right
            self.launcher_angle += 2.0
        self.launcher_angle = max(-170, min(-10, self.launcher_angle))

        # Launch ball on space press (edge trigger)
        # Only one ball can be active at a time for strategic play
        if space_held and not self.space_was_held and not self.active_balls:
            self._launch_ball()
        self.space_was_held = space_held

    def _launch_ball(self):
        if self.balls_remaining > 0:
            # sfx: launch_ball.wav
            self.balls_remaining -= 1
            angle_rad = math.radians(self.launcher_angle)
            start_pos = [
                self.launcher_pos[0] + self.LAUNCHER_LENGTH * math.cos(angle_rad),
                self.launcher_pos[1] + self.LAUNCHER_LENGTH * math.sin(angle_rad)
            ]
            velocity = [
                self.BALL_SPEED * math.cos(angle_rad),
                self.BALL_SPEED * math.sin(angle_rad)
            ]
            self.active_balls.append({'pos': start_pos, 'vel': velocity, 'radius': self.BALL_RADIUS})

    def _update_balls(self):
        step_reward = 0
        for ball in self.active_balls[:]:
            ball['pos'][0] += ball['vel'][0]
            ball['pos'][1] += ball['vel'][1]

            # Wall collisions
            if ball['pos'][0] - ball['radius'] < 0 or ball['pos'][0] + ball['radius'] > self.WIDTH:
                ball['vel'][0] *= -1
                ball['pos'][0] = np.clip(ball['pos'][0], ball['radius'], self.WIDTH - ball['radius'])
                # sfx: bounce_wall.wav
            if ball['pos'][1] - ball['radius'] < 0:
                ball['vel'][1] *= -1
                ball['pos'][1] = np.clip(ball['pos'][1], ball['radius'], self.HEIGHT - ball['radius'])
                # sfx: bounce_wall.wav

            # Ball goes off bottom of screen
            if ball['pos'][1] - ball['radius'] > self.HEIGHT:
                self.active_balls.remove(ball)
                continue

            # Block collisions
            for block in self.blocks[:]:
                if block['rect'].collidepoint(ball['pos']):
                    # sfx: block_hit.wav
                    self.score += block['value']
                    step_reward += 0.1 * block['value']
                    self._create_particles(block['rect'].center, self.BLOCK_COLORS[block['value']])
                    self.blocks.remove(block)
                    
                    # Simple bounce logic
                    # Determine if hit was more horizontal or vertical
                    dx = abs(ball['pos'][0] - block['rect'].centerx)
                    dy = abs(ball['pos'][1] - block['rect'].centery)
                    
                    if dx / block['rect'].width > dy / block['rect'].height:
                        ball['vel'][0] *= -1
                    else:
                        ball['vel'][1] *= -1
                    break # Only one block hit per frame
        return step_reward

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _check_termination(self):
        if not self.blocks and not self.win_condition:
            self.win_condition = True
            # sfx: win_level.wav
        
        if not self.loss_condition:
            if self.time_remaining <= 0:
                self.loss_condition = True
            if self.balls_remaining <= 0 and not self.active_balls:
                self.loss_condition = True
        
        if self.steps >= self.MAX_STEPS:
            return True

        return self.win_condition or self.loss_condition

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "balls_remaining": self.balls_remaining,
            "time_remaining_sec": self.time_remaining / self.FPS,
            "blocks_remaining": len(self.blocks),
        }

    def _render_background(self):
        for x in range(0, self.WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

    def _render_game(self):
        # Render particles
        for p in self.particles:
            alpha = max(0, 255 * (p['life'] / p['max_life']))
            color = (*p['color'], alpha)
            size = int(p['size'] * (p['life'] / p['max_life']))
            if size > 0:
                s = pygame.Surface((size, size), pygame.SRCALPHA)
                s.fill(color)
                self.screen.blit(s, (int(p['pos'][0] - size/2), int(p['pos'][1] - size/2)))

        # Render blocks
        for block in self.blocks:
            pygame.draw.rect(self.screen, self.BLOCK_COLORS[block['value']], block['rect'])
            pygame.draw.rect(self.screen, self.COLOR_BG, block['rect'], 1)

        # Render launcher base
        pygame.gfxdraw.filled_circle(self.screen, int(self.launcher_pos[0]), int(self.launcher_pos[1]), 15, self.COLOR_GRID)
        pygame.gfxdraw.aacircle(self.screen, int(self.launcher_pos[0]), int(self.launcher_pos[1]), 15, self.COLOR_PLAYER)

        # Render launcher barrel and aiming line
        angle_rad = math.radians(self.launcher_angle)
        end_x = self.launcher_pos[0] + self.LAUNCHER_LENGTH * math.cos(angle_rad)
        end_y = self.launcher_pos[1] + self.LAUNCHER_LENGTH * math.sin(angle_rad)
        pygame.draw.line(self.screen, self.COLOR_PLAYER, self.launcher_pos, (end_x, end_y), 4)

        # Aiming line (only if no balls are active)
        if not self.active_balls and not self.game_over:
            for i in range(1, 20):
                p_x = self.launcher_pos[0] + i * 20 * math.cos(angle_rad)
                p_y = self.launcher_pos[1] + i * 20 * math.sin(angle_rad)
                if p_y < 0: break
                pygame.draw.circle(self.screen, self.COLOR_PLAYER, (int(p_x), int(p_y)), 1)
        
        # Render balls
        for ball in self.active_balls:
            pygame.gfxdraw.filled_circle(self.screen, int(ball['pos'][0]), int(ball['pos'][1]), ball['radius'], self.COLOR_PLAYER)
            pygame.gfxdraw.aacircle(self.screen, int(ball['pos'][0]), int(ball['pos'][1]), ball['radius'], self.COLOR_PLAYER)

    def _render_ui(self):
        ui_surf = pygame.Surface((self.WIDTH, 40), pygame.SRCALPHA)
        ui_surf.fill(self.COLOR_UI_BG)
        self.screen.blit(ui_surf, (0, 0))

        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Balls
        balls_text = self.font_small.render(f"BALLS: {self.balls_remaining}", True, self.COLOR_TEXT)
        self.screen.blit(balls_text, (self.WIDTH // 2 - balls_text.get_width() // 2, 10))
        
        # Time
        time_sec = max(0, self.time_remaining / self.FPS)
        time_text = self.font_small.render(f"TIME: {time_sec:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(time_text, (self.WIDTH - time_text.get_width() - 10, 10))

        # Game Over / Win message
        if self.game_over:
            msg = "LEVEL CLEARED!" if self.win_condition else "GAME OVER"
            color = self.BLOCK_COLORS[1] if self.win_condition else self.BLOCK_COLORS[3]
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(end_text, text_rect)

    def _generate_blocks(self):
        self.blocks = []
        block_width, block_height = 40, 20
        rows, cols = 8, 14
        
        for r in range(rows):
            for c in range(cols):
                # Leave a gap in the middle for an initial clear shot
                if abs(c - cols // 2 + 0.5) < 2 and r > 4:
                    continue

                if self.np_random.random() > 0.3:
                    x = c * (block_width + 2) + 20
                    y = r * (block_height + 2) + 50
                    rect = pygame.Rect(x, y, block_width, block_height)
                    value = self.np_random.integers(1, 4) # 1, 2, or 3
                    self.blocks.append({'rect': rect, 'value': value})

    def _create_particles(self, pos, color, count=15):
        for _ in range(count):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * 2 + 1
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': self.np_random.integers(10, 20),
                'max_life': 20,
                'size': self.np_random.integers(3, 7),
                'color': color
            })
    
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    import os
    os.environ["SDL_VIDEODRIVER"] = "x11" # Use "x11", "dummy", or "windows" depending on your system

    env = GameEnv()
    obs, info = env.reset()
    
    # --- Pygame window setup for human play ---
    pygame.display.set_caption(env.game_description)
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    terminated = False
    while not terminated:
        # Map keyboard keys to the MultiDiscrete action space
        keys = pygame.key.get_pressed()
        
        movement_action = 0 # No-op
        if keys[pygame.K_LEFT]:
            movement_action = 3
        elif keys[pygame.K_RIGHT]:
            movement_action = 4
            
        space_action = 1 if keys[pygame.K_SPACE] else 0
        shift_action = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement_action, space_action, shift_action]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation from the environment to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
    
    print(f"Final Info: {info}")
    env.close()