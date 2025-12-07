
# Generated: 2025-08-28T05:31:58.573527
# Source Brief: brief_02653.md
# Brief Index: 2653

        
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
        "Controls: Press Space to jump. Avoid the red obstacles and reach the green flag."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A minimalist side-scrolling platformer. Time your jumps precisely to overcome obstacles and reach the end of each level."
    )

    # Frames auto-advance at 30fps.
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen dimensions
        self.WIDTH, self.HEIGHT = 640, 400

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
        
        # --- Game Constants ---
        self._setup_constants()
        
        # Initialize state variables
        self.player_pos = pygame.Vector2(0, 0)
        self.player_vel = pygame.Vector2(0, 0)
        self.is_jumping = False
        self.on_ground = False
        self.obstacles = []
        self.flag_rect = pygame.Rect(0, 0, 0, 0)
        self.particles = []
        
        self.level = 1
        self.total_score = 0
        self.level_timer = 0
        self.world_scroll_x = 0
        self.camera_y = 0
        
        self.steps = 0
        self.game_over = False

        # This call will fail if the implementation is incorrect.
        self.validate_implementation()

    def _setup_constants(self):
        # Colors
        self.COLOR_BG = (13, 20, 50) # Dark Blue
        self.COLOR_BG_ACCENT = (23, 30, 60)
        self.COLOR_PLAYER = (255, 255, 255) # White
        self.COLOR_OBSTACLE = (255, 50, 50) # Red
        self.COLOR_FLAG = (50, 255, 50) # Green
        self.COLOR_UI = (255, 215, 0) # Yellow/Gold
        self.COLOR_GROUND = (40, 50, 90)

        # Physics
        self.GRAVITY = 0.5
        self.JUMP_STRENGTH = -10
        self.PLAYER_X_POS = 100
        self.GROUND_Y = self.HEIGHT - 50
        self.SCROLL_SPEED = 5
        self.MAX_VEL_Y = 15

        # Player
        self.PLAYER_SIZE = 20
        
        # Level
        self.LEVEL_WIDTH_PIXELS = 5000
        self.BASE_OBSTACLES = 15
        self.LEVEL_TIME_LIMIT = 1800 # 60 seconds * 30 fps
        self.MAX_LEVELS = 3

        # UI
        self.font = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 18, bold=True)

    def _start_level(self):
        """Initializes the state for the current level."""
        self.player_pos = pygame.Vector2(self.PLAYER_X_POS, self.GROUND_Y - self.PLAYER_SIZE)
        self.player_vel = pygame.Vector2(0, 0)
        self.on_ground = True
        self.is_jumping = False
        
        self.world_scroll_x = 0
        self.camera_y = self.player_pos.y
        self.level_timer = self.LEVEL_TIME_LIMIT
        self.particles.clear()
        
        self._generate_level()

    def _generate_level(self):
        """Procedurally generates obstacles for the current level."""
        self.obstacles.clear()
        num_obstacles = int(self.BASE_OBSTACLES * (1 + 0.1 * (self.level - 1)))
        
        current_x = 600
        for _ in range(num_obstacles):
            gap = self.np_random.integers(150, 400)
            current_x += gap
            
            width = self.np_random.integers(30, 80)
            height = self.np_random.integers(20, 120)
            
            obstacle_rect = pygame.Rect(current_x, self.GROUND_Y - height, width, height)
            self.obstacles.append(obstacle_rect)
        
        flag_x = current_x + 500
        self.flag_rect = pygame.Rect(flag_x, self.GROUND_Y - 100, 15, 100)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.level = 1
        self.total_score = 0
        self.steps = 0
        self.game_over = False
        
        self._start_level()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0.1 # Survival reward
        self.steps += 1
        self.level_timer -= 1

        # 1. Handle Action
        self._handle_action(action)
        
        # 2. Update State
        self._update_player()
        self._update_world()

        # 3. Check for events and calculate rewards
        terminated = False
        
        # Obstacle collision
        if self._check_obstacle_collision():
            reward -= 50
            self.game_over = True
            terminated = True
        
        # Flag collision
        if self._check_flag_collision():
            if self.level < self.MAX_LEVELS:
                reward += 10
                self.total_score += 10
                self.level += 1
                self._start_level()
            else: # Final level complete
                reward += 100
                self.total_score += 100
                self.game_over = True
                terminated = True
        
        # Time out
        if self.level_timer <= 0:
            reward -= 10
            self.game_over = True
            terminated = True
            
        self.total_score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_action(self, action):
        space_pressed = action[1] == 1
        if space_pressed and self.on_ground:
            # sfx: Jump sound
            self.player_vel.y = self.JUMP_STRENGTH
            self.on_ground = False
            self.is_jumping = True

    def _update_player(self):
        # Apply gravity
        self.player_vel.y += self.GRAVITY
        self.player_vel.y = min(self.player_vel.y, self.MAX_VEL_Y)
        
        # Update position
        self.player_pos.y += self.player_vel.y
        
        # Ground collision
        if self.player_pos.y + self.PLAYER_SIZE >= self.GROUND_Y:
            self.player_pos.y = self.GROUND_Y - self.PLAYER_SIZE
            self.player_vel.y = 0
            if self.is_jumping: # Just landed
                # sfx: Landing thud
                self._create_landing_particles(5)
                self.is_jumping = False
            self.on_ground = True

    def _update_world(self):
        # Scroll world
        self.world_scroll_x += self.SCROLL_SPEED

        # Update particles
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

        # Smooth camera
        camera_target_y = self.player_pos.y - self.HEIGHT / 2.5
        self.camera_y += (camera_target_y - self.camera_y) * 0.1

    def _check_obstacle_collision(self):
        player_rect = pygame.Rect(self.player_pos.x, self.player_pos.y, self.PLAYER_SIZE, self.PLAYER_SIZE)
        for obs in self.obstacles:
            # Check against on-screen position
            obs_screen_rect = obs.move(-self.world_scroll_x, 0)
            if player_rect.colliderect(obs_screen_rect):
                # sfx: Collision/Fail sound
                return True
        return False

    def _check_flag_collision(self):
        player_rect = pygame.Rect(self.player_pos.x, self.player_pos.y, self.PLAYER_SIZE, self.PLAYER_SIZE)
        flag_screen_rect = self.flag_rect.move(-self.world_scroll_x, 0)
        if player_rect.colliderect(flag_screen_rect):
            # sfx: Level complete sound
            return True
        return False

    def _create_landing_particles(self, count):
        for _ in range(count):
            angle = self.np_random.uniform(math.pi, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            pos = self.player_pos + pygame.Vector2(self.PLAYER_SIZE / 2, self.PLAYER_SIZE)
            self.particles.append({'pos': pos, 'vel': vel, 'life': self.np_random.integers(10, 20)})
    
    def _get_observation(self):
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
        # --- Draw background elements ---
        # Parallax stars/dots
        for i in range(50):
            # Use a hash of i to get deterministic "random" positions
            hash_val = (i * 12345) % 10000 / 10000.0
            x = (hash_val * self.WIDTH * 5 - self.world_scroll_x * 0.5) % self.WIDTH
            y = ((hash_val * 7891) % 10000 / 10000.0 * self.HEIGHT)
            pygame.draw.rect(self.screen, self.COLOR_BG_ACCENT, (int(x), int(y), 2, 2))

        # --- Draw world elements relative to camera/scroll ---
        # Ground
        ground_rect = pygame.Rect(0, self.GROUND_Y - self.camera_y, self.WIDTH, self.HEIGHT - self.GROUND_Y)
        pygame.draw.rect(self.screen, self.COLOR_GROUND, ground_rect)

        # Obstacles
        for obs in self.obstacles:
            obs_screen_rect = obs.move(-self.world_scroll_x, -self.camera_y)
            if obs_screen_rect.right > 0 and obs_screen_rect.left < self.WIDTH:
                pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, obs_screen_rect)

        # Flag
        flag_screen_rect = self.flag_rect.move(-self.world_scroll_x, -self.camera_y)
        if flag_screen_rect.right > 0 and flag_screen_rect.left < self.WIDTH:
            pygame.draw.rect(self.screen, self.COLOR_FLAG, flag_screen_rect)
            pole_rect = pygame.Rect(flag_screen_rect.left, flag_screen_rect.bottom, 5, self.GROUND_Y - flag_screen_rect.bottom)
            pygame.draw.rect(self.screen, (200, 200, 200), pole_rect)

        # --- Draw player and effects ---
        # Player
        player_screen_pos = self.player_pos - pygame.Vector2(0, self.camera_y)
        player_rect = pygame.Rect(player_screen_pos.x, player_screen_pos.y, self.PLAYER_SIZE, self.PLAYER_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect)
        
        # Landing Particles
        for p in self.particles:
            p_screen_pos = p['pos'] - pygame.Vector2(self.PLAYER_X_POS, self.camera_y) + pygame.Vector2(self.player_pos.x, 0)
            size = max(1, int(p['life'] / 4))
            pygame.draw.rect(self.screen, self.COLOR_PLAYER, (int(p_screen_pos.x), int(p_screen_pos.y), size, size))

    def _render_ui(self):
        # Timer
        time_text = f"TIME: {self.level_timer // 30:02d}"
        time_surf = self.font.render(time_text, True, self.COLOR_UI)
        self.screen.blit(time_surf, (10, 10))

        # Level
        level_text = f"LEVEL: {self.level}/{self.MAX_LEVELS}"
        level_surf = self.font.render(level_text, True, self.COLOR_UI)
        self.screen.blit(level_surf, (self.WIDTH - level_surf.get_width() - 10, 10))

        # Score
        score_text = f"SCORE: {int(self.total_score)}"
        score_surf = self.font_small.render(score_text, True, self.COLOR_UI)
        self.screen.blit(score_surf, (10, 40))

        # Game Over message
        if self.game_over:
            win_condition = self.level == self.MAX_LEVELS and self._check_flag_collision()
            msg = "LEVELS COMPLETE!" if win_condition else "GAME OVER"
            
            over_surf = self.font.render(msg, True, self.COLOR_FLAG if win_condition else self.COLOR_OBSTACLE)
            over_rect = over_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            pygame.draw.rect(self.screen, (0,0,0,180), over_rect.inflate(20, 20))
            self.screen.blit(over_surf, over_rect)

    def _get_info(self):
        return {
            "score": self.total_score,
            "steps": self.steps,
            "level": self.level,
            "time_left": self.level_timer / 30,
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

# Example usage for testing
if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Minimalist Platformer")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        # Action defaults to no-op
        action = [0, 0, 0] # [movement, space, shift]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]:
            action[1] = 1

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Episode finished. Total Reward: {total_reward}, Info: {info}")
            # In a real scenario, you might wait for a keypress to reset
            # For this test, we'll just reset after a short pause.
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        clock.tick(30) # Match the environment's intended FPS

    env.close()