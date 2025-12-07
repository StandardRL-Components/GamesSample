# Generated: 2025-08-27T17:20:20.846984
# Source Brief: brief_01500.md
# Brief Index: 1500

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Press space for a small jump, hold shift for a large jump. Movement is automatic."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A side-scrolling arcade game where you control a space hopper. Jump over procedurally generated obstacles to reach the end of each level before time runs out."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    # Colors
    COLOR_BG_TOP = (20, 20, 40)
    COLOR_BG_BOTTOM = (0, 0, 0)
    COLOR_GROUND = (50, 200, 50)
    COLOR_PLAYER = (255, 60, 60)
    COLOR_PLAYER_GLOW = (255, 60, 60, 50)
    COLOR_OBSTACLE = (240, 240, 240)
    COLOR_PARTICLE = (255, 255, 100)
    COLOR_TEXT = (255, 255, 255)
    
    # Screen
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Game Physics & Params
    FPS = 30
    GRAVITY = 0.8
    JUMP_SMALL = 12
    JUMP_LARGE = 18
    HORIZONTAL_SPEED = 5
    PLAYER_RADIUS = 15
    PLAYER_SCREEN_X = SCREEN_WIDTH // 4
    GROUND_Y = SCREEN_HEIGHT - 40
    
    # Level Params
    LEVEL_LENGTH = 5000 # pixels
    LEVEL_TIME_SECONDS = 60
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 16)
        
        # Initialize state variables
        self.np_random = None
        self.hopper_y = 0
        self.hopper_vy = 0
        self.is_grounded = True
        self.world_x = 0
        self.obstacles = []
        self.cleared_obstacles = set()
        self.particles = []
        self.level = 1
        self.score = 0
        self.steps = 0
        self.time_remaining = 0
        self.game_over = False
        self.game_won = False

        self.level_configs = {
            1: {"density": 1.0 / 200.0, "min_gap": 50, "max_h": 60},
            2: {"density": 1.1 / 200.0, "min_gap": 47.5, "max_h": 80},
            3: {"density": 1.21 / 200.0, "min_gap": 45.125, "max_h": 100},
        }

        # self.reset() is called by the wrapper/runner, no need to call it here.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.level = 1
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.game_won = False
        self.particles = []
        
        self._reset_level_state()
        
        return self._get_observation(), self._get_info()

    def _reset_level_state(self):
        """Resets the state for the current level."""
        self.hopper_y = self.GROUND_Y
        self.hopper_vy = 0
        self.is_grounded = True
        self.world_x = 0
        self.time_remaining = self.LEVEL_TIME_SECONDS * self.FPS
        self._generate_level_obstacles()
        self.cleared_obstacles.clear()

    def _generate_level_obstacles(self):
        self.obstacles = []
        config = self.level_configs[self.level]
        # Start obstacles far enough away to pass stability test, which uses no-op actions.
        current_x = self.SCREEN_WIDTH
        
        while current_x < self.LEVEL_LENGTH - 200:
            if self.np_random.random() < config["density"] * 100:
                width = self.np_random.integers(20, 51)
                height = self.np_random.integers(20, config["max_h"] + 1)
                obstacle_rect = pygame.Rect(current_x, self.GROUND_Y - height, width, height)
                self.obstacles.append(obstacle_rect)
                current_x += width
            
            gap = self.np_random.integers(int(config["min_gap"]), int(config["min_gap"]) + 100)
            current_x += gap
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0.1 # Base reward for survival
        
        # Unpack factorized action
        # movement = action[0]  # Unused
        space_held = action[1] == 1
        shift_held = action[2] == 1
        
        # --- Update Game Logic ---
        self.steps += 1
        self.time_remaining -= 1
        
        # 1. Handle player jump action
        if self.is_grounded:
            if shift_held:
                self.hopper_vy = -self.JUMP_LARGE
                self.is_grounded = False
                # SFX: Large Jump
            elif space_held:
                self.hopper_vy = -self.JUMP_SMALL
                self.is_grounded = False
                # SFX: Small Jump

        # 2. Apply physics
        self.hopper_vy += self.GRAVITY
        self.hopper_y += self.hopper_vy
        
        # 3. Ground collision
        if self.hopper_y >= self.GROUND_Y:
            if not self.is_grounded:
                self._create_landing_particles(self.PLAYER_SCREEN_X + self.world_x, self.GROUND_Y)
                # SFX: Land
            self.hopper_y = self.GROUND_Y
            self.hopper_vy = 0
            self.is_grounded = True
        else:
            self.is_grounded = False

        # 4. Update world scroll
        self.world_x += self.HORIZONTAL_SPEED
        
        # 5. Update particles
        self._update_particles()
        
        # 6. Check for collisions and rewards
        player_world_x = self.world_x + self.PLAYER_SCREEN_X
        player_rect = pygame.Rect(
            player_world_x - self.PLAYER_RADIUS,
            self.hopper_y - self.PLAYER_RADIUS,
            self.PLAYER_RADIUS * 2,
            self.PLAYER_RADIUS * 2
        )

        for i, obs in enumerate(self.obstacles):
            if player_rect.colliderect(obs):
                reward = -100
                self.game_over = True
                # SFX: Crash
                break
            
            # Check for clearing an obstacle
            if i not in self.cleared_obstacles and player_rect.left > obs.right:
                self.cleared_obstacles.add(i)
                clearance = obs.top - player_rect.bottom
                if 0 <= clearance < 20:
                    reward += 1.0 # Risky jump
                elif clearance > 50:
                    reward -= 0.2 # Safe jump
        
        # 7. Check for level/game completion
        if not self.game_over and self.world_x >= self.LEVEL_LENGTH:
            self.score += 5
            reward += 5
            if self.level == 3:
                self.score += 100
                reward += 100
                self.game_over = True
                self.game_won = True
                # SFX: Victory
            else:
                self.level += 1
                self._reset_level_state()
                # SFX: Level Up
        
        # 8. Check for timeout
        if self.time_remaining <= 0 and not self.game_over:
            reward = -50
            self.game_over = True

        self.score += reward
        terminated = self.game_over
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _create_landing_particles(self, x, y):
        for _ in range(10):
            vel_x = self.np_random.uniform(-2, 2)
            vel_y = self.np_random.uniform(-3, -1)
            life = self.np_random.integers(10, 20)
            self.particles.append({'pos': [x, y], 'vel': [vel_x, vel_y], 'life': life})

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += self.GRAVITY * 0.1 # Lighter gravity for particles
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _get_observation(self):
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw gradient background
        for y in range(self.SCREEN_HEIGHT):
            ratio = y / self.SCREEN_HEIGHT
            color = (
                int(self.COLOR_BG_TOP[0] * (1 - ratio) + self.COLOR_BG_BOTTOM[0] * ratio),
                int(self.COLOR_BG_TOP[1] * (1 - ratio) + self.COLOR_BG_BOTTOM[1] * ratio),
                int(self.COLOR_BG_TOP[2] * (1 - ratio) + self.COLOR_BG_BOTTOM[2] * ratio)
            )
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))

        # Draw ground
        pygame.draw.line(self.screen, self.COLOR_GROUND, (0, self.GROUND_Y), (self.SCREEN_WIDTH, self.GROUND_Y), 3)

        # Draw obstacles
        for obs in self.obstacles:
            screen_x = obs.x - self.world_x
            if -obs.width < screen_x < self.SCREEN_WIDTH:
                pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, (screen_x, obs.y, obs.width, obs.height))
        
        # Draw particles
        for p in self.particles:
            screen_x = p['pos'][0] - self.world_x
            if 0 < screen_x < self.SCREEN_WIDTH:
                pygame.draw.circle(self.screen, self.COLOR_PARTICLE, (int(screen_x), int(p['pos'][1])), max(0, int(p['life'] / 4)))

        # Draw player
        player_pos_int = (int(self.PLAYER_SCREEN_X), int(self.hopper_y))
        # Glow effect
        glow_surf = pygame.Surface((self.PLAYER_RADIUS * 4, self.PLAYER_RADIUS * 4), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, self.COLOR_PLAYER_GLOW, (self.PLAYER_RADIUS * 2, self.PLAYER_RADIUS * 2), self.PLAYER_RADIUS + 5)
        self.screen.blit(glow_surf, (player_pos_int[0] - self.PLAYER_RADIUS * 2, player_pos_int[1] - self.PLAYER_RADIUS * 2))
        # Player circle
        pygame.gfxdraw.aacircle(self.screen, player_pos_int[0], player_pos_int[1], self.PLAYER_RADIUS, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_circle(self.screen, player_pos_int[0], player_pos_int[1], self.PLAYER_RADIUS, self.COLOR_PLAYER)

    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Level
        level_text = self.font_small.render(f"LEVEL: {self.level}/3", True, self.COLOR_TEXT)
        self.screen.blit(level_text, (self.SCREEN_WIDTH - level_text.get_width() - 10, 10))
        
        # Timer
        time_str = f"{max(0, self.time_remaining // self.FPS):02d}"
        timer_text = self.font_large.render(time_str, True, self.COLOR_TEXT)
        self.screen.blit(timer_text, (self.SCREEN_WIDTH // 2 - timer_text.get_width() // 2, 10))

        # Game Over / Victory Message
        if self.game_over:
            message = "GAME OVER"
            if self.game_won:
                message = "YOU WIN!"
            
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            self.screen.blit(overlay, (0, 0))
            
            end_text = self.font_large.render(message, True, self.COLOR_TEXT)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level,
            "time_remaining": self.time_remaining,
            "game_over": self.game_over
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
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example usage to test the environment visually
if __name__ == '__main__':
    # This block will fail in a headless environment. It is for local visual testing.
    os.environ["SDL_VIDEODRIVER"] = "x11"
    env = GameEnv()
    
    # --- Manual Play ---
    obs, info = env.reset()
    terminated = False
    
    # Pygame window for rendering
    pygame.display.set_caption("Space Hopper")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    while not terminated:
        # Action defaults
        movement = 0 # No-op
        space = 0
        shift = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]:
            space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift = 1
        if keys[pygame.K_q]:
             terminated = True
        
        action = np.array([movement, space, shift])
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(env.FPS)

    env.close()
    print("Game finished. Final Score:", info['score'])