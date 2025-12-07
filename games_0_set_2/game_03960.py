
# Generated: 2025-08-28T00:58:02.607176
# Source Brief: brief_03960.md
# Brief Index: 3960

        
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
        "Controls: Use arrow keys to aim your jump. Press SPACE to jump. Collect coins and reach the green goal platform."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Leap across procedurally generated platforms, collecting coins and risking daring jumps to reach the end."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 2000
        
        # Colors
        self.COLOR_BG_TOP = (10, 20, 40)
        self.COLOR_BG_BOTTOM = (0, 0, 0)
        self.COLOR_PLAYER = (255, 50, 50)
        self.COLOR_PLAYER_AIM = (255, 255, 255, 150)
        self.COLOR_PLATFORM = (220, 220, 220)
        self.COLOR_PLATFORM_BOUNCY = (100, 200, 255)
        self.COLOR_PLATFORM_SHORT = (200, 180, 100)
        self.COLOR_COIN = (255, 223, 0)
        self.COLOR_GOAL = (50, 255, 50)
        self.COLOR_PARTICLE = (255, 255, 255)
        self.COLOR_TEXT = (255, 255, 255)
        
        # Physics
        self.GRAVITY = 0.5
        self.MAX_FALL_SPEED = 10
        self.AIM_POWER_V_MIN = 8
        self.AIM_POWER_V_MAX = 16
        self.AIM_POWER_H_MAX = 8
        self.AIM_ADJUST_SPEED = 0.5
        self.BOUNCE_MULTIPLIER = 1.5

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
        self.font_main = pygame.font.SysFont("monospace", 20, bold=True)
        
        # Internal state variables (initialized in reset)
        self.steps = None
        self.score = None
        self.game_over = None
        self.player_pos = None
        self.player_vel = None
        self.on_ground = None
        self.jump_aim = None
        self.prev_space_held = None
        self.platforms = None
        self.coins = None
        self.particles = None
        self.camera_y = None
        self.highest_platform_y = None
        self.last_player_y = None
        self.platform_gap_increase = None
        self.step_reward = None

        # Initialize state variables
        self.reset()
        
        # Run validation
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT - 50)
        self.player_vel = pygame.Vector2(0, 0)
        self.last_player_y = self.player_pos.y
        
        self.on_ground = True
        self.jump_aim = pygame.Vector2(0, self.AIM_POWER_V_MIN)
        self.prev_space_held = False

        self.platforms = []
        self.coins = []
        self.particles = []
        
        self.camera_y = 0
        self.platform_gap_increase = 0
        self.step_reward = 0

        # Create initial platforms
        start_platform = self._create_platform(
            pygame.Rect(self.WIDTH/2 - 50, self.HEIGHT - 30, 100, 20), 'standard'
        )
        self.platforms.append(start_platform)
        self.highest_platform_y = start_platform['rect'].top
        self._generate_initial_platforms()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.clock.tick(self.FPS)
        self.steps += 1
        self.step_reward = 0

        # Unpack factorized action
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        self._handle_input(movement, space_held)
        self._update_player()
        self._handle_collisions()
        self._update_particles()
        self._update_camera()
        self._manage_world_generation()
        
        # Calculate reward for vertical progress
        y_diff = self.last_player_y - self.player_pos.y
        if y_diff > 0:
            self.step_reward += y_diff * 0.1
        self.last_player_y = self.player_pos.y
        
        terminated = self._check_termination()
        
        return (
            self._get_observation(),
            self.step_reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_held):
        # Adjust aim vector
        if self.on_ground:
            if movement == 1: # Up
                self.jump_aim.y = min(self.AIM_POWER_V_MAX, self.jump_aim.y + self.AIM_ADJUST_SPEED)
            elif movement == 2: # Down
                self.jump_aim.y = max(self.AIM_POWER_V_MIN, self.jump_aim.y - self.AIM_ADJUST_SPEED)
            elif movement == 3: # Left
                self.jump_aim.x = max(-self.AIM_POWER_H_MAX, self.jump_aim.x - self.AIM_ADJUST_SPEED)
            elif movement == 4: # Right
                self.jump_aim.x = min(self.AIM_POWER_H_MAX, self.jump_aim.x + self.AIM_ADJUST_SPEED)
        
        # Jump on space press
        if space_held and not self.prev_space_held and self.on_ground:
            self.player_vel.x = self.jump_aim.x
            self.player_vel.y = -self.jump_aim.y
            self.on_ground = False
            # sfx: jump
        
        self.prev_space_held = space_held

    def _update_player(self):
        if not self.on_ground:
            self.player_vel.y += self.GRAVITY
            self.player_vel.y = min(self.player_vel.y, self.MAX_FALL_SPEED)
        
        self.player_pos += self.player_vel

        # Keep player horizontally centered in the world logic
        self.player_pos.x = self.WIDTH / 2
        
    def _handle_collisions(self):
        self.on_ground = False
        player_rect = pygame.Rect(self.player_pos.x - 5, self.player_pos.y - 10, 10, 10)

        # Platform collision
        if self.player_vel.y >= 0:
            for p in self.platforms:
                if player_rect.colliderect(p['rect']) and player_rect.bottom < p['rect'].centery:
                    self.player_pos.y = p['rect'].top
                    self.player_vel.y = 0
                    self.on_ground = True
                    self._create_particles(self.player_pos, 10)
                    
                    if p['type'] == 'bouncy':
                        self.player_vel.y = -self.jump_aim.y * self.BOUNCE_MULTIPLIER
                        self.on_ground = False
                        self.step_reward += 2
                        self.score += 2
                        # sfx: bounce
                    elif p['type'] == 'short':
                        self.step_reward -= 0.2
                    else: # standard
                        # sfx: land
                        pass
                    break
        
        # Coin collection
        for coin in self.coins[:]:
            if player_rect.colliderect(coin['rect']):
                self.coins.remove(coin)
                self.score += 1
                self.step_reward += 1
                self._create_particles(coin['rect'].center, 5, self.COLOR_COIN)
                # sfx: coin_collect

    def _update_camera(self):
        # Camera scrolls up to keep player in the middle third of the screen
        target_camera_y = self.player_pos.y - self.HEIGHT * 0.6
        self.camera_y += (target_camera_y - self.camera_y) * 0.1

    def _manage_world_generation(self):
        # Difficulty scaling
        if self.steps > 0 and self.steps % 200 == 0:
            self.platform_gap_increase += 0.2
            
        # Generate new platforms if needed
        while self.highest_platform_y > self.camera_y:
            self._generate_new_platform()
            
        # Prune old entities
        lowest_y = self.camera_y + self.HEIGHT + 50
        self.platforms = [p for p in self.platforms if p['rect'].top > lowest_y or p['type'] == 'goal']
        self.coins = [c for c in self.coins if c['rect'].top > lowest_y]

    def _generate_initial_platforms(self):
        for _ in range(20):
            self._generate_new_platform()
        
        # Add goal platform
        goal_y = self.highest_platform_y - 200
        goal_platform = self._create_platform(pygame.Rect(self.WIDTH/2 - 75, goal_y, 150, 30), 'goal')
        self.platforms.append(goal_platform)

    def _generate_new_platform(self):
        max_gap = 60 + self.platform_gap_increase
        min_gap = 30 + self.platform_gap_increase
        
        y_pos = self.highest_platform_y - self.np_random.integers(min_gap, max_gap)
        x_pos = self.np_random.integers(50, self.WIDTH - 150)
        
        # Determine platform type
        rand_type = self.np_random.random()
        if rand_type < 0.1: # 10% bouncy
            p_type = 'bouncy'
            width = self.np_random.integers(60, 90)
        elif rand_type < 0.25: # 15% short
            p_type = 'short'
            width = self.np_random.integers(30, 50)
        else: # 75% standard
            p_type = 'standard'
            width = self.np_random.integers(80, 120)

        new_platform = self._create_platform(pygame.Rect(x_pos, y_pos, width, 20), p_type)
        self.platforms.append(new_platform)
        self.highest_platform_y = y_pos

        # Chance to spawn a coin on the platform
        if self.np_random.random() < 0.4 and p_type != 'short':
            coin_pos_x = x_pos + width / 2
            coin_pos_y = y_pos - 20
            self.coins.append({'rect': pygame.Rect(coin_pos_x - 5, coin_pos_y - 5, 10, 10)})

    def _create_platform(self, rect, p_type):
        return {'rect': rect, 'type': p_type}

    def _check_termination(self):
        # Reached goal
        player_rect = pygame.Rect(self.player_pos.x - 5, self.player_pos.y - 10, 10, 10)
        for p in self.platforms:
            if p['type'] == 'goal' and player_rect.colliderect(p['rect']):
                self.step_reward += 100
                self.score += 100
                self.game_over = True
                return True
        
        # Fell off screen
        if self.player_pos.y > self.camera_y + self.HEIGHT + 50:
            self.step_reward -= 10
            self.game_over = True
            return True
        
        # Max steps
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
            
        return False

    def _get_observation(self):
        self._draw_gradient_background()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
        
    def _draw_gradient_background(self):
        for y in range(self.HEIGHT):
            ratio = y / self.HEIGHT
            color = (
                int(self.COLOR_BG_TOP[0] * (1 - ratio) + self.COLOR_BG_BOTTOM[0] * ratio),
                int(self.COLOR_BG_TOP[1] * (1 - ratio) + self.COLOR_BG_BOTTOM[1] * ratio),
                int(self.COLOR_BG_TOP[2] * (1 - ratio) + self.COLOR_BG_BOTTOM[2] * ratio),
            )
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

    def _render_game(self):
        # Draw platforms
        for p in self.platforms:
            screen_rect = p['rect'].copy()
            screen_rect.y -= int(self.camera_y)
            color = self.COLOR_PLATFORM
            if p['type'] == 'bouncy': color = self.COLOR_PLATFORM_BOUNCY
            elif p['type'] == 'short': color = self.COLOR_PLATFORM_SHORT
            elif p['type'] == 'goal': color = self.COLOR_GOAL
            pygame.draw.rect(self.screen, color, screen_rect, border_radius=3)
            pygame.draw.rect(self.screen, tuple(c*0.7 for c in color), screen_rect, width=2, border_radius=3)

        # Draw coins
        for coin in self.coins:
            screen_pos = (int(coin['rect'].centerx), int(coin['rect'].centery - self.camera_y))
            pygame.gfxdraw.filled_circle(self.screen, screen_pos[0], screen_pos[1], 7, self.COLOR_COIN)
            pygame.gfxdraw.aacircle(self.screen, screen_pos[0], screen_pos[1], 7, self.COLOR_COIN)

        # Draw particles
        for particle in self.particles:
            pos = (int(particle['pos'].x), int(particle['pos'].y - self.camera_y))
            alpha = max(0, 255 * (particle['life'] / particle['max_life']))
            color = (*particle['color'], alpha)
            temp_surf = pygame.Surface((4, 4), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (2, 2), 2)
            self.screen.blit(temp_surf, (pos[0] - 2, pos[1] - 2))

        # Draw player
        player_screen_pos = (int(self.player_pos.x), int(self.player_pos.y - self.camera_y))
        player_rect = pygame.Rect(player_screen_pos[0] - 7, player_screen_pos[1] - 14, 14, 14)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=2)
        
        # Draw aim indicator
        if self.on_ground:
            aim_end_x = player_screen_pos[0] + self.jump_aim.x * 3
            aim_end_y = player_screen_pos[1] - self.jump_aim.y * 2
            pygame.draw.line(self.screen, self.COLOR_PLAYER_AIM, player_screen_pos, (aim_end_x, aim_end_y), 2)
            pygame.draw.circle(self.screen, self.COLOR_PLAYER_AIM, (int(aim_end_x), int(aim_end_y)), 3)

    def _render_ui(self):
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        steps_text = self.font_main.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (self.WIDTH - steps_text.get_width() - 10, 10))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_y": self.player_pos.y,
            "on_ground": self.on_ground,
        }

    def _create_particles(self, pos, count, color=None):
        if color is None: color = self.COLOR_PARTICLE
        for _ in range(count):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * 2 + 1
            vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            life = self.np_random.integers(10, 20)
            self.particles.append({'pos': pos.copy(), 'vel': vel, 'life': life, 'max_life': life, 'color': color})
            
    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['vel'] *= 0.95
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def validate_implementation(self):
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
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    # --- To play manually ---
    # This requires a windowed display
    try:
        screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        pygame.display.set_caption("Hopper")
        
        obs, info = env.reset()
        done = False
        
        while not done:
            # Map keyboard keys to the MultiDiscrete action space
            keys = pygame.key.get_pressed()
            
            movement = 0 # none
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = [movement, space_held, shift_held]
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Render the observation from the environment to the screen
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
            
            if done:
                print(f"Game Over! Final Score: {info['score']}, Steps: {info['steps']}")
                # Wait a moment before resetting
                pygame.time.wait(2000)
                obs, info = env.reset()
                done = False

    finally:
        pygame.quit()