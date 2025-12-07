
# Generated: 2025-08-27T20:22:14.862570
# Source Brief: brief_02434.md
# Brief Index: 2434

        
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
        "Controls: ←→ to aim, ↑↓ to adjust power. Press Space to launch. Hold Shift to reset aim."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "An arcade physics game. Launch your projectile to destroy all the red target bricks within 4 shots. "
        "Consecutive hits in one shot grant bonus points."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 1000
        self.TOTAL_SHOTS = 4
        self.NUM_TARGET_BRICKS = 20
        self.NUM_OBSTACLE_BRICKS = 5

        # Colors
        self.COLOR_BG = (25, 35, 45)
        self.COLOR_LAUNCHER = (230, 230, 240)
        self.COLOR_AIM_LINE = (255, 200, 0)
        self.COLOR_TARGET = (220, 50, 50)
        self.COLOR_TARGET_DESTROYED = (80, 140, 80)
        self.COLOR_OBSTACLE = (100, 110, 120)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_WIN = (100, 255, 100)
        self.COLOR_LOSE = (255, 100, 100)

        # Physics
        self.GRAVITY = 0.15
        self.MIN_ANGLE = math.pi * 0.1
        self.MAX_ANGLE = math.pi * 0.9
        self.MIN_POWER = 5.0
        self.MAX_POWER = 15.0
        self.POWER_STEP = 0.2
        self.ANGLE_STEP = 0.03

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 64)
        self.font_medium = pygame.font.Font(None, 32)
        
        # Initialize state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_phase = "aiming"
        self.launcher_pos = (self.WIDTH // 2, self.HEIGHT - 30)
        self.launch_angle = 0
        self.launch_power = 0
        self.shots_remaining = 0
        self.bricks = []
        self.particles = []
        self.trajectory_points = []
        self.prev_space_held = False

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_phase = "aiming"
        self.shots_remaining = self.TOTAL_SHOTS
        self.particles = []
        self.prev_space_held = False
        
        self._generate_bricks()
        self._reset_aim()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        reward = 0
        terminated = False
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        if self.game_phase == "aiming":
            launch_triggered = self._handle_aiming(movement, space_held, shift_held)
            if launch_triggered:
                self.shots_remaining -= 1
                # sound: launch_projectile.wav
                reward += self._simulate_flight()
                self.game_phase = "aiming" # Return to aiming after flight
                self._calculate_trajectory() # Recalculate for next shot

        self.steps += 1
        
        # Check for termination conditions
        targets_destroyed = all(b['is_destroyed'] for b in self.bricks if b['is_target'])
        
        if targets_destroyed:
            reward += 100  # Victory bonus
            self.game_over = True
            terminated = True
            self.game_phase = "win"
        elif self.shots_remaining <= 0 and self.game_phase == "aiming":
            reward -= 100  # Loss penalty
            self.game_over = True
            terminated = True
            self.game_phase = "lose"
        
        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
            self.game_phase = "lose"

        self.prev_space_held = space_held
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_aiming(self, movement, space_held, shift_held):
        if shift_held:
            self._reset_aim()
            return False

        if movement == 1: # Up
            self.launch_power = min(self.MAX_POWER, self.launch_power + self.POWER_STEP)
        elif movement == 2: # Down
            self.launch_power = max(self.MIN_POWER, self.launch_power - self.POWER_STEP)
        elif movement == 3: # Left
            self.launch_angle = min(self.MAX_ANGLE, self.launch_angle + self.ANGLE_STEP)
        elif movement == 4: # Right
            self.launch_angle = max(self.MIN_ANGLE, self.launch_angle - self.ANGLE_STEP)
        
        if movement != 0 or shift_held:
            self._calculate_trajectory()
        
        return space_held and not self.prev_space_held

    def _simulate_flight(self):
        total_reward = 0
        hits_this_shot = 0
        
        vx = self.launch_power * math.cos(self.launch_angle)
        vy = -self.launch_power * math.sin(self.launch_angle)
        proj_pos = list(self.launcher_pos)
        
        for _ in range(300): # Max flight time
            proj_pos[0] += vx
            proj_pos[1] += vy
            vy += self.GRAVITY
            
            # Check for wall collisions
            if not (0 < proj_pos[0] < self.WIDTH and 0 < proj_pos[1] < self.HEIGHT):
                break

            # Check for brick collisions
            collided = False
            for brick in self.bricks:
                if not brick['is_destroyed']:
                    brick_rect = pygame.Rect(brick['pos'][0], brick['pos'][1], brick['size'][0], brick['size'][1])
                    if brick_rect.collidepoint(proj_pos):
                        total_reward += 0.1 # Small reward for any hit
                        if brick['is_target']:
                            brick['is_destroyed'] = True
                            self.score += 10 # Base score for a target
                            total_reward += 1.0
                            hits_this_shot += 1
                            if hits_this_shot > 1:
                                self.score += 20 * (hits_this_shot - 1) # Combo bonus
                                total_reward += 2.0 * (hits_this_shot - 1)
                            # sound: brick_destroy.wav
                            self._create_explosion(brick['pos'], self.COLOR_TARGET)
                        else: # Hit an obstacle
                            # sound: obstacle_hit.wav
                            self._create_explosion(brick['pos'], self.COLOR_OBSTACLE)

                        collided = True
                        break # Projectile is destroyed on first hit
            if collided:
                break
        return total_reward

    def _generate_bricks(self):
        self.bricks = []
        brick_w, brick_h = 30, 15
        grid_w, grid_h = 18, 10
        x_offset = (self.WIDTH - grid_w * (brick_w + 5)) // 2
        y_offset = 50

        available_pos = []
        for row in range(grid_h):
            for col in range(grid_w):
                available_pos.append((
                    x_offset + col * (brick_w + 5),
                    y_offset + row * (brick_h + 5)
                ))
        
        self.np_random.shuffle(available_pos)

        for i in range(self.NUM_TARGET_BRICKS):
            pos = available_pos.pop(0)
            self.bricks.append({
                'pos': pos, 'size': (brick_w, brick_h), 'is_target': True, 
                'is_destroyed': False, 'color': self.COLOR_TARGET
            })
        
        for i in range(self.NUM_OBSTACLE_BRICKS):
            pos = available_pos.pop(0)
            self.bricks.append({
                'pos': pos, 'size': (brick_w, brick_h), 'is_target': False, 
                'is_destroyed': False, 'color': self.COLOR_OBSTACLE
            })

    def _reset_aim(self):
        self.launch_angle = math.pi / 2
        self.launch_power = (self.MIN_POWER + self.MAX_POWER) / 2
        self._calculate_trajectory()

    def _calculate_trajectory(self):
        self.trajectory_points = []
        vx = self.launch_power * math.cos(self.launch_angle)
        vy = -self.launch_power * math.sin(self.launch_angle)
        pos = list(self.launcher_pos)
        for _ in range(40): # Number of points in the trajectory line
            pos[0] += vx * 0.5
            pos[1] += vy * 0.5
            vy += self.GRAVITY * 0.5
            self.trajectory_points.append(tuple(pos))

    def _create_explosion(self, pos, color):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = self.np_random.integers(15, 30)
            self.particles.append({'pos': list(pos), 'vel': vel, 'lifespan': lifespan, 'color': color})

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw particles
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)
            else:
                alpha = max(0, min(255, int(255 * (p['lifespan'] / 30))))
                size = max(1, int(3 * (p['lifespan'] / 30)))
                pygame.draw.circle(self.screen, p['color'] + (alpha,), [int(c) for c in p['pos']], size)

        # Draw bricks
        for brick in self.bricks:
            color = self.COLOR_TARGET_DESTROYED if (brick['is_target'] and brick['is_destroyed']) else brick['color']
            pygame.draw.rect(self.screen, color, (*brick['pos'], *brick['size']), border_radius=3)
            pygame.draw.rect(self.screen, tuple(c*0.7 for c in color), (*brick['pos'], *brick['size']), 1, border_radius=3)

        # Draw launcher
        pygame.draw.circle(self.screen, self.COLOR_LAUNCHER, self.launcher_pos, 10)
        pygame.draw.circle(self.screen, self.COLOR_BG, self.launcher_pos, 7)

        # Draw aiming line
        if self.game_phase == "aiming" and len(self.trajectory_points) > 1:
            pygame.draw.aalines(self.screen, self.COLOR_AIM_LINE, False, self.trajectory_points)
            
    def _render_ui(self):
        # Score
        score_text = self.font_medium.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 10))

        # Shots
        shots_text = self.font_medium.render(f"Shots: {self.shots_remaining}", True, self.COLOR_TEXT)
        self.screen.blit(shots_text, (10, 10))
        
        # Power/Angle indicator
        if self.game_phase == "aiming":
            power_percent = (self.launch_power - self.MIN_POWER) / (self.MAX_POWER - self.MIN_POWER)
            angle_deg = math.degrees(self.launch_angle)
            
            power_bar_w = 100
            power_bar_h = 10
            power_fill_w = int(power_bar_w * power_percent)
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, (10, self.HEIGHT - 20, power_bar_w, power_bar_h))
            pygame.draw.rect(self.screen, self.COLOR_AIM_LINE, (10, self.HEIGHT - 20, power_fill_w, power_bar_h))
            
            angle_text = self.font_medium.render(f"{180-angle_deg:.0f}°", True, self.COLOR_TEXT)
            self.screen.blit(angle_text, (120, self.HEIGHT - 25))

        # Game over message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            if self.game_phase == "win":
                msg_text = self.font_large.render("YOU WIN!", True, self.COLOR_WIN)
            else: # lose
                msg_text = self.font_large.render("GAME OVER", True, self.COLOR_LOSE)
                
            text_rect = msg_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(msg_text, text_rect)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "shots_remaining": self.shots_remaining,
            "targets_destroyed": sum(1 for b in self.bricks if b['is_target'] and b['is_destroyed']),
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        print("Beginning implementation validation...")
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


if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Set up a window to display the game
    pygame.display.set_caption(env.game_description)
    display_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    terminated = False
    
    # Game loop
    while not terminated:
        movement, space, shift = 0, 0, 0
        
        # Pygame event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)

        # Update the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Game Over! Final Score: {info['score']}")
            pygame.time.wait(3000) # Wait 3 seconds before closing
            
        env.clock.tick(30) # Limit to 30 FPS for human play

    env.close()