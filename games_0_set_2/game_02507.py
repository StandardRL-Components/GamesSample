
# Generated: 2025-08-28T05:04:41.377083
# Source Brief: brief_02507.md
# Brief Index: 2507

        
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
        "Controls: Press Space to jump over the red obstacles. Timing is everything!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Synth Sprint: A retro-futuristic rhythm runner. Jump in sync with the beat to build your combo and reach the finish line."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and World Dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        self.WORLD_LENGTH = 12000 # Corresponds to roughly 1000 steps at max speed

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
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 64)
        self.font_popup = pygame.font.Font(None, 32)
        
        # Colors
        self.COLOR_BG = (10, 5, 30)
        self.COLOR_GRID_BG = (15, 15, 50)
        self.COLOR_GRID_FG = (20, 20, 80)
        self.COLOR_PLAYER = (0, 255, 255)
        self.COLOR_PLAYER_GLOW = (0, 128, 128)
        self.COLOR_OBSTACLE = (255, 0, 100)
        self.COLOR_OBSTACLE_GLOW = (128, 0, 50)
        self.COLOR_FINISH = (0, 255, 128)
        self.COLOR_TEXT = (255, 255, 200)
        self.COLOR_GROUND = (40, 40, 120)

        # Player Physics & State
        self.PLAYER_X = self.WIDTH // 4
        self.PLAYER_WIDTH, self.PLAYER_HEIGHT = 20, 20
        self.GROUND_Y = self.HEIGHT - 60
        self.GRAVITY = 0.8
        self.JUMP_STRENGTH = -15
        
        # Game Mechanics
        self.MAX_STEPS = 1000
        self.INITIAL_OBSTACLE_SPEED = 6.0
        
        # Initialize state variables
        self.player_y = 0
        self.player_vy = 0
        self.is_grounded = True
        self.world_progress = 0.0
        self.obstacles = []
        self.particles = []
        self.reward_popups = []
        self.combo = 1
        self.obstacle_speed = 0.0
        self.obstacle_spawn_timer = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # Initialize state variables properly
        self.reset()

        # Run validation check
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.world_progress = 0.0
        
        # Reset player
        self.player_y = self.GROUND_Y - self.PLAYER_HEIGHT
        self.player_vy = 0
        self.is_grounded = True

        # Reset mechanics
        self.obstacles = []
        self.particles = []
        self.reward_popups = []
        self.combo = 1
        self.obstacle_speed = self.INITIAL_OBSTACLE_SPEED
        self.obstacle_spawn_timer = 30 # Spawn first obstacle quickly

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        # Unpack factorized action
        space_held = action[1] == 1

        reward = 0.1 # Survival reward
        
        # --- GAME LOGIC ---
        
        # Player jump
        if space_held and self.is_grounded:
            self.player_vy = self.JUMP_STRENGTH
            self.is_grounded = False
            # sfx: jump_sound()
            self._create_particles(self.PLAYER_X + self.PLAYER_WIDTH / 2, self.GROUND_Y, self.COLOR_PLAYER, 15, 'up')

        # Player physics
        self.player_vy += self.GRAVITY
        self.player_y += self.player_vy
        
        if self.player_y >= self.GROUND_Y - self.PLAYER_HEIGHT:
            if not self.is_grounded:
                # sfx: land_sound()
                self._create_particles(self.PLAYER_X + self.PLAYER_WIDTH / 2, self.GROUND_Y, self.COLOR_GROUND, 10, 'side')
            self.player_y = self.GROUND_Y - self.PLAYER_HEIGHT
            self.player_vy = 0
            self.is_grounded = True

        # World progression
        self.world_progress += self.obstacle_speed

        # Difficulty scaling
        if self.steps > 0 and self.steps % 500 == 0:
            self.obstacle_speed += 0.5 # Simplified from 0.05/frame as this is per-step

        # Obstacle management
        self._update_obstacles()
        
        # Particle and popup management
        self._update_particles_and_popups()

        # Collision and reward checking
        player_rect = pygame.Rect(self.PLAYER_X, self.player_y, self.PLAYER_WIDTH, self.PLAYER_HEIGHT)
        for obs in self.obstacles:
            obs_x = obs['x'] - self.world_progress
            obs_rect = pygame.Rect(obs_x, obs['y'], obs['w'], obs['h'])
            
            # Collision check
            if player_rect.colliderect(obs_rect):
                self.game_over = True
                # sfx: crash_sound()
                self._create_particles(player_rect.centerx, player_rect.centery, self.COLOR_OBSTACLE, 50, 'explosion')
                break

            # Obstacle clear check
            if not obs['cleared'] and obs_rect.right < player_rect.left:
                obs['cleared'] = True
                # sfx: clear_sound()
                
                reward += 1.0
                
                # Check for perfect timing
                perfect_jump_min_y = self.GROUND_Y - self.PLAYER_HEIGHT - 90
                perfect_jump_max_y = self.GROUND_Y - self.PLAYER_HEIGHT - 60
                
                if perfect_jump_min_y < self.player_y < perfect_jump_max_y:
                    # Perfect Jump
                    perfect_reward = 2.0 * self.combo
                    reward += perfect_reward
                    self.score += perfect_reward
                    self.combo += 1
                    self.reward_popups.append({'text': f'PERFECT! x{self.combo}', 'pos': [self.PLAYER_X, self.player_y - 40], 'life': 40, 'color': self.COLOR_FINISH})
                else:
                    # Safe Jump
                    safe_penalty = 0.2 * self.combo
                    reward -= safe_penalty
                    self.score -= safe_penalty
                    self.combo = 1
                    self.reward_popups.append({'text': 'OK', 'pos': [self.PLAYER_X, self.player_y - 40], 'life': 30, 'color': self.COLOR_TEXT})

        self.score += reward
        terminated = self._check_termination()

        # Apply terminal rewards
        if self.game_over:
            reward = -50
            self.score += reward
        elif self.world_progress >= self.WORLD_LENGTH:
            reward = 100
            self.score += reward

        self.steps += 1
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_obstacles(self):
        self.obstacle_spawn_timer -= 1
        if self.obstacle_spawn_timer <= 0:
            obstacle_x = self.world_progress + self.WIDTH + 50
            obstacle_h = random.randint(30, 60)
            self.obstacles.append({
                'x': obstacle_x,
                'y': self.GROUND_Y - obstacle_h,
                'w': 25,
                'h': obstacle_h,
                'cleared': False
            })
            min_interval = max(20, 60 - self.steps // 50)
            max_interval = max(40, 100 - self.steps // 50)
            self.obstacle_spawn_timer = random.randint(min_interval, max_interval)
            
        # Remove off-screen obstacles
        self.obstacles = [obs for obs in self.obstacles if obs['x'] - self.world_progress > -obs['w']]
        
    def _update_particles_and_popups(self):
        # Update particles
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

        # Update popups
        for pop in self.reward_popups:
            pop['pos'][1] -= 0.5
            pop['life'] -= 1
        self.reward_popups = [pop for pop in self.reward_popups if pop['life'] > 0]

    def _check_termination(self):
        return self.game_over or self.steps >= self.MAX_STEPS or self.world_progress >= self.WORLD_LENGTH

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
            "combo": self.combo,
            "world_progress": self.world_progress,
        }

    def _render_game(self):
        self._render_background()
        self._render_finish_line()
        self._render_particles()
        self._render_obstacles()
        self._render_player()
        self._render_reward_popups()
        
    def _render_background(self):
        # Far background grid (slower scroll)
        for i in range(0, self.WIDTH + 100, 50):
            x = i - int(self.world_progress * 0.2) % 50
            pygame.draw.line(self.screen, self.COLOR_GRID_BG, (x, 0), (x, self.HEIGHT), 1)
        for i in range(0, self.HEIGHT, 50):
            pygame.draw.line(self.screen, self.COLOR_GRID_BG, (0, i), (self.WIDTH, i), 1)

        # Near background grid (faster scroll)
        for i in range(0, self.WIDTH + 100, 100):
            x = i - int(self.world_progress * 0.5) % 100
            pygame.draw.line(self.screen, self.COLOR_GRID_FG, (x, 0), (x, self.HEIGHT), 2)
        for i in range(0, self.HEIGHT, 100):
            pygame.draw.line(self.screen, self.COLOR_GRID_FG, (0, i), (self.WIDTH, i), 2)

    def _render_finish_line(self):
        finish_x = self.WORLD_LENGTH - self.world_progress
        if finish_x < self.WIDTH:
            pygame.draw.rect(self.screen, self.COLOR_FINISH, (finish_x, 0, 20, self.HEIGHT))
            for i in range(0, self.HEIGHT, 20):
                color1 = (0,0,0) if (i // 20) % 2 == 0 else self.COLOR_FINISH
                color2 = self.COLOR_FINISH if (i // 20) % 2 == 0 else (0,0,0)
                pygame.draw.rect(self.screen, color1, (finish_x, i, 10, 20))
                pygame.draw.rect(self.screen, color2, (finish_x + 10, i, 10, 20))
                
    def _render_obstacles(self):
        for obs in self.obstacles:
            x = obs['x'] - self.world_progress
            rect = pygame.Rect(x, obs['y'], obs['w'], obs['h'])
            
            # Glow effect
            glow_rect = rect.inflate(10, 10)
            glow_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(glow_surf, self.COLOR_OBSTACLE_GLOW + (80,), (0, 0, glow_rect.width, glow_rect.height), border_radius=5)
            self.screen.blit(glow_surf, glow_rect.topleft)

            # Main shape
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, rect, border_radius=3)
            # Highlight
            pygame.draw.line(self.screen, (255, 150, 200), rect.topleft, rect.topright, 2)

    def _render_player(self):
        player_rect = pygame.Rect(self.PLAYER_X, self.player_y, self.PLAYER_WIDTH, self.PLAYER_HEIGHT)
        
        # Glow effect
        glow_size = max(self.PLAYER_WIDTH, 40 + int(self.player_vy)) # Dynamic glow on jump
        glow_surf = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(glow_surf, glow_size // 2, glow_size // 2, glow_size // 2, self.COLOR_PLAYER_GLOW + (100,))
        self.screen.blit(glow_surf, (player_rect.centerx - glow_size // 2, player_rect.centery - glow_size // 2))

        # Main shape with squash/stretch
        squash = min(5, max(-5, self.player_vy))
        stretch = min(5, max(-5, -self.player_vy))
        
        display_rect = pygame.Rect(
            self.PLAYER_X - squash,
            self.player_y - stretch,
            self.PLAYER_WIDTH + 2 * squash,
            self.PLAYER_HEIGHT + 2 * stretch
        )
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, display_rect, border_radius=3)

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color = p['color'] + (alpha,)
            radius = int(p['radius'] * (p['life'] / p['max_life']))
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), radius, color)

    def _render_reward_popups(self):
        for pop in self.reward_popups:
            alpha = int(255 * (pop['life'] / 40))
            text_surf = self.font_popup.render(pop['text'], True, pop['color'])
            text_surf.set_alpha(alpha)
            text_rect = text_surf.get_rect(center=(pop['pos'][0], pop['pos'][1]))
            self.screen.blit(text_surf, text_rect)

    def _render_ui(self):
        # Ground
        pygame.draw.rect(self.screen, self.COLOR_GROUND, (0, self.GROUND_Y, self.WIDTH, self.HEIGHT - self.GROUND_Y))
        pygame.draw.line(self.screen, self.COLOR_PLAYER, (0, self.GROUND_Y), (self.WIDTH, self.GROUND_Y), 3)

        # Combo display
        combo_text = f"COMBO x{self.combo}"
        combo_surf = self.font_large.render(combo_text, True, self.COLOR_TEXT)
        self.screen.blit(combo_surf, (20, 20))
        
        # Score display
        score_text = f"SCORE: {int(self.score)}"
        score_surf = self.font_small.render(score_text, True, self.COLOR_TEXT)
        score_rect = score_surf.get_rect(topright=(self.WIDTH - 20, 20))
        self.screen.blit(score_surf, score_rect)

        # Progress bar
        progress_pct = min(1.0, self.world_progress / self.WORLD_LENGTH)
        bar_width = self.WIDTH - 40
        pygame.draw.rect(self.screen, self.COLOR_GRID_FG, (20, self.HEIGHT - 30, bar_width, 10))
        pygame.draw.rect(self.screen, self.COLOR_FINISH, (20, self.HEIGHT - 30, bar_width * progress_pct, 10))

    def _create_particles(self, x, y, color, count, p_type):
        for _ in range(count):
            if p_type == 'explosion':
                vel = [random.uniform(-5, 5), random.uniform(-5, 5)]
            elif p_type == 'up':
                vel = [random.uniform(-1, 1), random.uniform(-5, -1)]
            elif p_type == 'side':
                 vel = [random.choice([-1, 1]) * random.uniform(1, 3), random.uniform(-1, 0)]
            else: # default
                vel = [0,0]

            life = random.randint(20, 40)
            self.particles.append({
                'pos': [x, y],
                'vel': vel,
                'life': life,
                'max_life': life,
                'color': color,
                'radius': random.randint(2, 5)
            })

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