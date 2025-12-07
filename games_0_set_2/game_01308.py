
# Generated: 2025-08-27T16:43:40.256442
# Source Brief: brief_01308.md
# Brief Index: 1308

        
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
    user_guide = "Controls: Use arrow keys to move. Survive the horde for 60 seconds."

    # Must be a short, user-facing description of the game:
    game_description = "A fast-paced, top-down arcade game where you must dodge a horde of zombies to survive for 60 seconds."

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    FPS = 30
    GAME_DURATION_SECONDS = 60
    MAX_STEPS = GAME_DURATION_SECONDS * FPS

    # Colors
    COLOR_BG = (20, 25, 40)
    COLOR_WALL = (80, 80, 100)
    COLOR_PLAYER = (50, 255, 50)
    COLOR_PLAYER_GLOW = (150, 255, 150, 50)
    COLOR_ZOMBIE = (255, 50, 50)
    COLOR_ZOMBIE_GLOW = (255, 100, 100, 40)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_WIN_TEXT = (100, 255, 100)
    COLOR_LOSE_TEXT = (255, 100, 100)

    # Game Area
    GAME_AREA_PADDING = 20
    GAME_AREA_X = GAME_AREA_PADDING
    GAME_AREA_Y = GAME_AREA_PADDING
    GAME_AREA_WIDTH = SCREEN_WIDTH - 2 * GAME_AREA_PADDING
    GAME_AREA_HEIGHT = SCREEN_HEIGHT - 2 * GAME_AREA_PADDING

    # Entity properties
    PLAYER_SIZE = 12
    PLAYER_SPEED = 4.0
    ZOMBIE_SIZE = 10
    ZOMBIE_COUNT = 5
    ZOMBIE_SPEED = 1.5
    
    # Rewards
    REWARD_STEP = 0.1
    REWARD_WIN = 100.0
    REWARD_LOSE = -100.0

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 72)
        self.font_medium = pygame.font.Font(None, 48)
        
        # Initialize state variables
        self.player_pos = [0, 0]
        self.zombies = []
        self.particles = []
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.game_won = False
        self.time_survived = 0.0
        self.random_generator = random.Random()

        self.reset()
        
        # self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.random_generator = random.Random(seed)

        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.game_won = False
        self.time_survived = 0.0

        # Place player in the center
        self.player_pos = [
            self.GAME_AREA_X + self.GAME_AREA_WIDTH / 2,
            self.GAME_AREA_Y + self.GAME_AREA_HEIGHT / 2,
        ]

        # Spawn zombies
        self.zombies = []
        min_spawn_dist_sq = (self.GAME_AREA_WIDTH / 4) ** 2
        for _ in range(self.ZOMBIE_COUNT):
            while True:
                x = self.random_generator.uniform(self.GAME_AREA_X, self.GAME_AREA_X + self.GAME_AREA_WIDTH)
                y = self.random_generator.uniform(self.GAME_AREA_Y, self.GAME_AREA_Y + self.GAME_AREA_HEIGHT)
                dist_sq = (x - self.player_pos[0])**2 + (y - self.player_pos[1])**2
                if dist_sq > min_spawn_dist_sq:
                    self.zombies.append(pygame.Rect(x, y, self.ZOMBIE_SIZE, self.ZOMBIE_SIZE))
                    break
        
        self.particles = []

        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)
            self.time_survived += 1 / self.FPS

        terminated = False
        reward = 0

        if not self.game_over:
            self.steps += 1
            
            # --- Update Game Logic ---
            self._handle_input(action)
            self._update_zombies()
            self._update_particles()

            # --- Collision and Termination Checks ---
            if self._check_player_collision():
                # sfx: player_death_explosion
                self.game_over = True
                terminated = True
                reward = self.REWARD_LOSE
                self._create_particle_burst(self.player_pos, self.COLOR_PLAYER, 50)
            elif self.steps >= self.MAX_STEPS:
                # sfx: win_jingle
                self.game_over = True
                self.game_won = True
                terminated = True
                reward = self.REWARD_WIN
            else:
                reward = self.REWARD_STEP

        self.score += reward

        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_input(self, action):
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        # space_held and shift_held are ignored as per the brief
        
        dx, dy = 0, 0
        if movement == 1: dy = -1  # Up
        elif movement == 2: dy = 1   # Down
        elif movement == 3: dx = -1  # Left
        elif movement == 4: dx = 1   # Right
        
        if dx != 0 or dy != 0:
            # sfx: player_footstep (subtle)
            # Create a small trail of particles
            if self.steps % 3 == 0:
                self._create_particle_burst(self.player_pos, self.COLOR_PLAYER, 1, speed_mult=0.5, lifespan=10)

        self.player_pos[0] += dx * self.PLAYER_SPEED
        self.player_pos[1] += dy * self.PLAYER_SPEED

        # Clamp player position to game area
        self.player_pos[0] = max(self.GAME_AREA_X, min(self.player_pos[0], self.GAME_AREA_X + self.GAME_AREA_WIDTH - self.PLAYER_SIZE))
        self.player_pos[1] = max(self.GAME_AREA_Y, min(self.player_pos[1], self.GAME_AREA_Y + self.GAME_AREA_HEIGHT - self.PLAYER_SIZE))

    def _update_zombies(self):
        for zombie in self.zombies:
            # sfx: zombie_groan (ambient)
            dx = self.player_pos[0] - zombie.centerx
            dy = self.player_pos[1] - zombie.centery
            dist = math.hypot(dx, dy)

            if dist > 1:
                zombie.x += (dx / dist) * self.ZOMBIE_SPEED
                zombie.y += (dy / dist) * self.ZOMBIE_SPEED

    def _check_player_collision(self):
        player_rect = pygame.Rect(self.player_pos[0], self.player_pos[1], self.PLAYER_SIZE, self.PLAYER_SIZE)
        for zombie in self.zombies:
            if player_rect.colliderect(zombie):
                return True
        return False
        
    def _create_particle_burst(self, pos, color, count, speed_mult=1.0, lifespan=20):
        for _ in range(count):
            angle = self.random_generator.uniform(0, 2 * math.pi)
            speed = self.random_generator.uniform(1, 4) * speed_mult
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            particle_lifespan = self.random_generator.randint(lifespan // 2, lifespan)
            self.particles.append({'pos': list(pos), 'vel': vel, 'life': particle_lifespan, 'max_life': particle_lifespan, 'color': color})

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][0] *= 0.95 # friction
            p['vel'][1] *= 0.95 # friction
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw game area boundary
        pygame.draw.rect(self.screen, self.COLOR_WALL, (self.GAME_AREA_X, self.GAME_AREA_Y, self.GAME_AREA_WIDTH, self.GAME_AREA_HEIGHT), 3)

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            size = int(self.PLAYER_SIZE/3 * (p['life'] / p['max_life']))
            if size > 0:
                pygame.gfxdraw.filled_circle(
                    self.screen, int(p['pos'][0] + self.PLAYER_SIZE/2), int(p['pos'][1] + self.PLAYER_SIZE/2),
                    size,
                    (p['color'][0], p['color'][1], p['color'][2], alpha)
                )

        # Draw zombies
        for zombie in self.zombies:
            pygame.gfxdraw.filled_circle(self.screen, int(zombie.centerx), int(zombie.centery), int(self.ZOMBIE_SIZE * 1.5), self.COLOR_ZOMBIE_GLOW)
            pygame.draw.rect(self.screen, self.COLOR_ZOMBIE, zombie)

        # Draw player, unless they have just lost
        if not (self.game_over and not self.game_won):
            player_center = (int(self.player_pos[0] + self.PLAYER_SIZE / 2), int(self.player_pos[1] + self.PLAYER_SIZE / 2))
            pygame.gfxdraw.filled_circle(self.screen, player_center[0], player_center[1], int(self.PLAYER_SIZE * 1.8), self.COLOR_PLAYER_GLOW)
            player_rect = pygame.Rect(self.player_pos[0], self.player_pos[1], self.PLAYER_SIZE, self.PLAYER_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect)
            
    def _render_ui(self):
        # Draw timer
        time_left = max(0, self.GAME_DURATION_SECONDS - self.time_survived)
        timer_text = f"TIME: {time_left:.1f}"
        text_surface = self.font_medium.render(timer_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(text_surface, (self.SCREEN_WIDTH - text_surface.get_width() - 15, 10))

        # Draw game over/win message
        if self.game_over:
            if self.game_won:
                msg = "YOU SURVIVED!"
                color = self.COLOR_WIN_TEXT
            else:
                msg = "GAME OVER"
                color = self.COLOR_LOSE_TEXT
            
            text_surface = self.font_large.render(msg, True, color)
            text_rect = text_surface.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(text_surface, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
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
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game directly
    import os
    # For headless execution, uncomment the next line
    # os.environ["SDL_VIDEODRIVER"] = "dummy"

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # --- Pygame setup for human play ---
    human_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Zombie Survival")
    
    running = True
    total_reward = 0
    
    while running:
        # --- Human input mapping to action space ---
        movement = 0 # No-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
            movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [movement, space_held, shift_held]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                total_reward = 0

        # --- Step the environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Info: {info}")
            
        # --- Render the observation to the screen ---
        # The observation is (H, W, C), but pygame surfaces are (W, H)
        # So we need to transpose it back for displaying
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        human_screen.blit(surf, (0, 0))
        pygame.display.flip()

    env.close()