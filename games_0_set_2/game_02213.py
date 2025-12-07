
# Generated: 2025-08-28T04:05:27.091825
# Source Brief: brief_02213.md
# Brief Index: 2213

        
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
        "Controls: ←→ to run, ↑ to jump. Collect yellow coins and reach the white finish line."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Control a robot in a fast-paced, side-scrolling platformer. Jump over red obstacles, collect coins, and race to the finish."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and world dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        self.LEVEL_LENGTH_PIXELS = 4000
        self.GROUND_Y = 350

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
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 48, bold=True)

        # Colors
        self.COLOR_BG = (15, 15, 25) # Dark blue/black
        self.COLOR_GROUND = (40, 40, 50)
        self.COLOR_PLAYER = (60, 160, 255) # Bright Blue
        self.COLOR_OBSTACLE = (255, 50, 100) # Bright Red
        self.COLOR_COIN = (255, 220, 0) # Bright Yellow
        self.COLOR_FINISH = (255, 255, 255)
        self.COLOR_UI = (240, 240, 240)
        
        # Player physics
        self.PLAYER_SPEED = 5
        self.GRAVITY = 0.8
        self.JUMP_STRENGTH = 15

        # Game state variables (initialized in reset)
        self.steps = None
        self.score = None
        self.game_over = None
        self.player_pos = None
        self.player_vel = None
        self.is_jumping = None
        self.camera_x = None
        self.obstacles = None
        self.coins = None
        self.finish_line_x = None
        self.max_progress = None
        self.obstacle_speed = None
        self.parallax_stars = None

        self.reset()
        
        # This check is for development and ensures compliance
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False

        self.player_pos = np.array([100.0, self.GROUND_Y - 40.0])
        self.player_vel = np.array([0.0, 0.0])
        self.is_jumping = True
        self.max_progress = self.player_pos[0]

        self.camera_x = 0
        self.finish_line_x = self.LEVEL_LENGTH_PIXELS
        self.obstacle_speed = 1.0
        
        self._generate_level()
        self._generate_background()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            # If the game is over, do nothing but return the final state
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        terminated = False

        # 1. Handle Input
        self._handle_input(action)

        # 2. Update Physics
        self._update_physics()

        # 3. Check Collisions & Collectibles
        reward += self._check_collisions()
        
        # 4. Update Difficulty
        if self.steps > 0 and self.steps % 500 == 0:
            self.obstacle_speed += 0.05

        # 5. Calculate Progress Reward
        progress = self.player_pos[0] - self.max_progress
        if progress > 0:
            reward += progress * 0.1
            self.max_progress = self.player_pos[0]

        # 6. Check Termination Conditions
        if self.player_pos[0] >= self.finish_line_x:
            reward += 100  # Goal-oriented reward for finishing
            terminated = True
            self.game_over = True
            # // play victory sound
        
        if self.steps >= 2000:
            terminated = True
            self.game_over = True

        self.steps += 1
        
        if terminated and reward > 0: # If won, ensure score reflects it
             pass # reward is already added
        elif terminated and reward <= 0: # If lost (obstacle or timeout)
            if reward == 0: # Timeout case
                reward = -10 # Small penalty for timeout
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _handle_input(self, action):
        movement = action[0]
        
        # Horizontal movement
        if movement == 3:  # Left
            self.player_vel[0] = -self.PLAYER_SPEED
        elif movement == 4:  # Right
            self.player_vel[0] = self.PLAYER_SPEED
        else:
            self.player_vel[0] = 0

        # Jumping
        if movement == 1 and not self.is_jumping:  # Up
            self.player_vel[1] = -self.JUMP_STRENGTH
            self.is_jumping = True
            # // play jump sound

    def _update_physics(self):
        # Apply gravity
        self.player_vel[1] += self.GRAVITY
        
        # Update position
        self.player_pos += self.player_vel

        # Ground collision
        player_bottom = self.player_pos[1] + 40
        if player_bottom >= self.GROUND_Y:
            self.player_pos[1] = self.GROUND_Y - 40
            self.player_vel[1] = 0
            if self.is_jumping:
                self.is_jumping = False
                # // play landing sound

        # Keep player from moving off the left edge of the camera
        self.player_pos[0] = max(self.player_pos[0], self.camera_x)

    def _check_collisions(self):
        reward = 0
        player_rect = pygame.Rect(self.player_pos[0], self.player_pos[1], 30, 40)

        # Obstacles
        for obstacle in self.obstacles:
            if player_rect.colliderect(obstacle['rect']):
                reward -= 50  # Event-based penalty for collision
                self.game_over = True
                # // play collision sound
                break
        
        # Coins
        for coin in self.coins:
            if not coin['collected']:
                coin_rect = pygame.Rect(coin['pos'][0]-5, coin['pos'][1]-5, 10, 10)
                if player_rect.colliderect(coin_rect):
                    coin['collected'] = True
                    self.score += 1
                    reward += 5  # Event-based reward for collecting coin
                    # // play coin collect sound
        return reward

    def _generate_level(self):
        self.obstacles = []
        self.coins = []
        current_x = 500

        while current_x < self.finish_line_x - 400:
            gap = self.np_random.integers(120, 250)
            current_x += gap
            
            choice = self.np_random.random()
            if choice < 0.6: # Place an obstacle
                height = self.np_random.integers(20, 80)
                width = self.np_random.integers(30, 60)
                self.obstacles.append({
                    'rect': pygame.Rect(current_x, self.GROUND_Y - height, width, height)
                })
                current_x += width
            elif choice < 0.9: # Place coins
                num_coins = self.np_random.integers(3, 6)
                coin_y = self.np_random.integers(self.GROUND_Y - 150, self.GROUND_Y - 50)
                for i in range(num_coins):
                    self.coins.append({
                        'pos': [current_x + i * 30, coin_y],
                        'collected': False
                    })
                current_x += num_coins * 30

    def _generate_background(self):
        self.parallax_stars = {
            'near': [],
            'mid': [],
            'far': []
        }
        for _ in range(50):
            self.parallax_stars['far'].append(
                (self.np_random.integers(0, self.LEVEL_LENGTH_PIXELS), self.np_random.integers(0, self.GROUND_Y))
            )
        for _ in range(30):
            self.parallax_stars['mid'].append(
                (self.np_random.integers(0, self.LEVEL_LENGTH_PIXELS), self.np_random.integers(0, self.GROUND_Y))
            )
        for _ in range(15):
            self.parallax_stars['near'].append(
                (self.np_random.integers(0, self.LEVEL_LENGTH_PIXELS), self.np_random.integers(0, self.GROUND_Y))
            )

    def _get_observation(self):
        # Update camera to keep player centered
        self.camera_x = self.player_pos[0] - self.WIDTH / 2.2

        # Clear screen with background
        self.screen.fill(self.COLOR_BG)

        # Render all game elements
        self._render_game()

        # Render UI overlay
        self._render_ui()

        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Parallax Background
        for x, y in self.parallax_stars['far']:
            screen_x = (x - self.camera_x * 0.2) % self.WIDTH
            pygame.gfxdraw.pixel(self.screen, int(screen_x), int(y), (50, 50, 70))
        for x, y in self.parallax_stars['mid']:
            screen_x = (x - self.camera_x * 0.5) % self.WIDTH
            pygame.draw.circle(self.screen, (90, 90, 110), (int(screen_x), int(y)), 1)
        for x, y in self.parallax_stars['near']:
            screen_x = (x - self.camera_x * 0.8) % self.WIDTH
            pygame.draw.circle(self.screen, (150, 150, 170), (int(screen_x), int(y)), 2)

        # Ground
        pygame.draw.rect(self.screen, self.COLOR_GROUND, (0, self.GROUND_Y, self.WIDTH, self.HEIGHT - self.GROUND_Y))

        # Finish Line
        finish_screen_x = self.finish_line_x - self.camera_x
        if 0 < finish_screen_x < self.WIDTH:
            pygame.draw.line(self.screen, self.COLOR_FINISH, (finish_screen_x, 0), (finish_screen_x, self.GROUND_Y), 5)

        # Coins
        coin_anim_frame = (self.steps // 6) % 4
        anim_widths = [10, 6, 2, 6]
        coin_width = anim_widths[coin_anim_frame]
        for coin in self.coins:
            if not coin['collected']:
                screen_x = coin['pos'][0] - self.camera_x
                if 0 < screen_x < self.WIDTH:
                    pygame.draw.ellipse(self.screen, self.COLOR_COIN, pygame.Rect(screen_x - coin_width/2, coin['pos'][1] - 5, coin_width, 10))
                    # Glow effect
                    pygame.draw.ellipse(self.screen, (255, 255, 100, 50), pygame.Rect(screen_x - 10, coin['pos'][1] - 10, 20, 20), 1)

        # Obstacles
        for obstacle in self.obstacles:
            screen_rect = obstacle['rect'].copy()
            screen_rect.x -= self.camera_x
            if screen_rect.colliderect(self.screen.get_rect()):
                pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, screen_rect)
                pygame.draw.rect(self.screen, (255, 150, 180), screen_rect, 2) # Highlight

        # Player
        self._render_player()

    def _render_player(self):
        player_screen_pos = self.player_pos - np.array([self.camera_x, 0])
        px, py = int(player_screen_pos[0]), int(player_screen_pos[1])
        
        # Body
        body_rect = pygame.Rect(px, py, 30, 40)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, body_rect, border_radius=4)
        
        # Eye
        eye_x = px + (20 if self.player_vel[0] >= 0 else 10)
        pygame.draw.circle(self.screen, (255, 255, 255), (eye_x, py + 12), 4)
        pygame.draw.circle(self.screen, (0, 0, 0), (eye_x, py + 12), 2)
        
        # Legs animation
        leg_length = 12
        if self.is_jumping:
            # Tucked legs
            pygame.draw.line(self.screen, self.COLOR_PLAYER, (px + 10, py + 40), (px + 10, py + 40 + leg_length/2), 6)
            pygame.draw.line(self.screen, self.COLOR_PLAYER, (px + 20, py + 40), (px + 20, py + 40 + leg_length/2), 6)
        else:
            # Running animation
            anim_phase = (self.steps // 4) % 4
            angle1 = math.sin(anim_phase * math.pi / 2) * 30
            angle2 = math.sin((anim_phase + 2) * math.pi / 2) * 30
            
            leg1_end = (px + 10 + math.sin(math.radians(angle1)) * leg_length, py + 40 + math.cos(math.radians(angle1)) * leg_length)
            leg2_end = (px + 20 + math.sin(math.radians(angle2)) * leg_length, py + 40 + math.cos(math.radians(angle2)) * leg_length)
            
            pygame.draw.line(self.screen, self.COLOR_PLAYER, (px + 10, py + 40), leg1_end, 6)
            pygame.draw.line(self.screen, self.COLOR_PLAYER, (px + 20, py + 40), leg2_end, 6)

    def _render_ui(self):
        # Time elapsed
        time_text = f"TIME: {self.steps / 30:.1f}s"
        time_surf = self.font_ui.render(time_text, True, self.COLOR_UI)
        self.screen.blit(time_surf, (10, 10))

        # Coins collected
        coin_text = f"COINS: {self.score}"
        coin_surf = self.font_ui.render(coin_text, True, self.COLOR_COIN)
        self.screen.blit(coin_surf, (self.WIDTH - coin_surf.get_width() - 10, 10))

        # Game Over message
        if self.game_over:
            message = "FINISH!" if self.player_pos[0] >= self.finish_line_x else "GAME OVER"
            color = (100, 255, 100) if message == "FINISH!" else self.COLOR_OBSTACLE
            
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            self.screen.blit(overlay, (0, 0))

            end_surf = self.font_game_over.render(message, True, color)
            end_rect = end_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_surf, end_rect)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "progress_percent": round(self.player_pos[0] / self.finish_line_x * 100, 2),
        }
    
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation.
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

# Example of how to run the environment
if __name__ == "__main__":
    import os
    os.environ["SDL_VIDEODRIVER"] = "x11" # Use 'x11', 'dummy' or 'windib' depending on your system

    env = GameEnv(render_mode="rgb_array")
    
    # --- To play the game manually ---
    # `pip install getkey` to use this manual player
    try:
        from getkey import getkey, keys
    except ImportError:
        print("Install 'getkey' to play manually: pip install getkey")
        getkey = None

    if getkey:
        obs, info = env.reset()
        terminated = False
        total_reward = 0
        
        print("Manual Control:")
        print(env.user_guide)
        
        # Pygame window for human play
        pygame.display.set_caption("Robot Runner")
        human_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))

        while not terminated:
            # Map keyboard to MultiDiscrete action
            action = [0, 0, 0] # [movement, space, shift]
            
            key = getkey(blocking=False)
            if key == keys.UP:
                action[0] = 1
            elif key == keys.DOWN:
                action[0] = 2
            elif key == keys.LEFT:
                action[0] = 3
            elif key == keys.RIGHT:
                action[0] = 4
            elif key == keys.SPACE:
                action[1] = 1
            elif key == keys.SHIFT:
                action[2] = 1
            elif key == 'q':
                 break

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            # Render for human
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            human_screen.blit(surf, (0, 0))
            pygame.display.flip()

            env.clock.tick(30) # Limit to 30 FPS

            if terminated:
                print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
                # Wait a bit before closing
                pygame.time.wait(2000)

    env.close()