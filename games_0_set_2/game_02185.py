
# Generated: 2025-08-28T04:00:13.140283
# Source Brief: brief_02185.md
# Brief Index: 2185

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A minimalist side-scrolling arcade game where the player must precisely time 
    jumps to ascend through procedurally generated levels filled with obstacles.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = "Controls: Press space to jump and ascend. Avoid the red obstacles."

    # Must be a short, user-facing description of the game:
    game_description = "A minimalist arcade platformer. Time your jumps precisely to climb through procedurally generated levels and reach the top."

    # Frames auto-advance for real-time physics and smooth graphics.
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    LEVEL_HEIGHT = 1200  # Vertical world distance per level

    # Colors
    COLOR_BG_TOP = (10, 0, 30)
    COLOR_BG_BOTTOM = (0, 0, 0)
    COLOR_PLAYER = (0, 255, 150)
    COLOR_OBSTACLE_TRI = (255, 80, 80)
    COLOR_OBSTACLE_RECT = (255, 150, 50)
    COLOR_OBSTACLE_CIRCLE = (255, 200, 0)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_PARTICLE_JUMP = (200, 255, 220)
    COLOR_PARTICLE_WIN = (255, 255, 100)

    # Physics and Gameplay
    GRAVITY = 0.3
    JUMP_STRENGTH = -8.5
    MAX_VEL_Y = 10
    PLAYER_SIZE = 20
    MAX_STEPS = 5000
    WIN_LEVEL = 10
    OBSTACLE_COUNT = 25

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.render_mode = render_mode
        
        # EXACT spaces required by the brief
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup for headless rendering
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 40, bold=True)
        
        self.background_surf = self._create_background()
        
        # State variables are initialized in reset()
        self.player_world_y = None
        self.player_vel_y = None
        self.player_screen_pos = None
        self.camera_y = None
        self.obstacles = None
        self.particles = None
        self.score = None
        self.steps = None
        self.level = None
        self.game_over = None
        self.win = None
        self.space_pressed_last_frame = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Player state
        self.player_world_y = 0
        self.player_vel_y = 0
        self.player_screen_pos = [self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT * 0.8]
        
        # Game state
        self.camera_y = self.player_world_y - self.player_screen_pos[1]
        self.obstacles = []
        self.particles = []
        self.score = 0
        self.steps = 0
        self.level = 1
        self.game_over = False
        self.win = False
        self.space_pressed_last_frame = False
        
        for _ in range(self.OBSTACLE_COUNT):
            self._spawn_obstacle(initial_spawn=True)
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        space_pressed = action[1] == 1
        reward = 0.1  # Survival reward

        # --- Player Logic ---
        if space_pressed and not self.space_pressed_last_frame:
            self.player_vel_y = self.JUMP_STRENGTH
            # Sound: Player Jump
            self._spawn_particles(self.player_screen_pos[0], self.player_screen_pos[1] + self.PLAYER_SIZE / 2, 10, self.COLOR_PARTICLE_JUMP, is_jump_burst=True)

            risk_box = pygame.Rect(self.player_screen_pos[0] - 75, self.player_screen_pos[1] - 75, 150, 150)
            is_risky = any(obs['screen_rect'].colliderect(risk_box) for obs in self.obstacles if 'screen_rect' in obs)
            reward += 2.0 if is_risky else -0.2

        self.space_pressed_last_frame = space_pressed
        
        self.player_vel_y += self.GRAVITY
        self.player_vel_y = min(self.player_vel_y, self.MAX_VEL_Y)
        self.player_world_y += self.player_vel_y

        # --- Camera & Entity Updates ---
        target_camera_y = self.player_world_y - self.SCREEN_HEIGHT * 2 / 3
        self.camera_y += (target_camera_y - self.camera_y) * 0.08
        self.player_screen_pos[1] = self.player_world_y - self.camera_y

        self._update_obstacles()
        self._update_particles()
        
        # --- Progression & Termination ---
        new_level = max(1, 1 + int(-self.player_world_y / self.LEVEL_HEIGHT))
        if new_level > self.level:
            self.level = new_level
            reward += 1.0
            self.score += 100
            # Sound: Level Up
            self._spawn_particles(self.player_screen_pos[0], self.player_screen_pos[1], 30, self.COLOR_PARTICLE_WIN)

        terminated = self._check_termination()
        if terminated:
            if self.win:
                reward = 100.0
                self.score += 1000
                # Sound: Victory Fanfare
                self._spawn_particles(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2, 100, self.COLOR_PARTICLE_WIN)
            else:
                reward = -10.0
                self.score = max(0, self.score - 50)
                # Sound: Explosion / Player Death
                player_rect = self._get_player_rect()
                self._spawn_particles(player_rect.centerx, player_rect.centery, 50, self.COLOR_PLAYER)
                self._spawn_particles(player_rect.centerx, player_rect.centery, 20, (255, 255, 255))
        
        self.score += reward
        self.score = max(0, self.score)

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _check_termination(self):
        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True

        if self.level >= self.WIN_LEVEL and not self.win:
            self.win = True
            self.game_over = True
            return True

        player_rect = self._get_player_rect()
        if any(player_rect.colliderect(obs['screen_rect']) for obs in self.obstacles if 'screen_rect' in obs):
            self.game_over = True
            return True

        if player_rect.top > self.SCREEN_HEIGHT + 50: # Fell off bottom
            self.game_over = True
            return True
            
        return False

    def _update_obstacles(self):
        speed_multiplier = 1 + (self.level - 1) * 0.05
        
        active_obstacles = []
        for obs in self.obstacles:
            obs['rect'].x += obs['vel_x'] * speed_multiplier
            if obs['type'] == 'circle':
                obs['rect'].y += math.sin(self.steps * 0.1 + obs['phase']) * 2
            
            is_on_screen = (obs['vel_x'] > 0 and obs['rect'].left < self.SCREEN_WIDTH) or \
                           (obs['vel_x'] < 0 and obs['rect'].right > 0)
            if is_on_screen:
                active_obstacles.append(obs)
        self.obstacles = active_obstacles

        while len(self.obstacles) < self.OBSTACLE_COUNT:
             self._spawn_obstacle()

    def _spawn_obstacle(self, initial_spawn=False):
        y_pos = self.camera_y + self.np_random.uniform(-self.SCREEN_HEIGHT * 0.5, self.SCREEN_HEIGHT * 1.5)
        if initial_spawn:
            y_pos = self.np_random.uniform(self.player_world_y - self.LEVEL_HEIGHT, self.player_world_y)

        side = self.np_random.choice([0, 1])
        obs_type = self.np_random.choice(['triangle', 'rect', 'circle'])
        size = self.np_random.uniform(20, 60)
        
        x_pos, vel_x = (-size, self.np_random.uniform(1, 3)) if side == 0 else (self.SCREEN_WIDTH, self.np_random.uniform(-3, -1))
        rect = pygame.Rect(x_pos, y_pos, size, size if obs_type != 'rect' else size * 0.6)
        
        if obs_type == 'rect': color, vel_mod = self.COLOR_OBSTACLE_RECT, 1.2
        elif obs_type == 'circle': color, vel_mod = self.COLOR_OBSTACLE_CIRCLE, 1.5
        else: color, vel_mod = self.COLOR_OBSTACLE_TRI, 1.0

        self.obstacles.append({
            'rect': rect, 'type': obs_type, 'vel_x': vel_x * vel_mod,
            'color': color, 'phase': self.np_random.uniform(0, 2 * math.pi)
        })

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1
            p['life'] -= 1

    def _spawn_particles(self, x, y, count, color, is_jump_burst=False):
        for _ in range(count):
            angle = self.np_random.uniform(0.25 * math.pi, 0.75 * math.pi) if is_jump_burst else self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3) if is_jump_burst else self.np_random.uniform(1, 5)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = self.np_random.integers(20, 40)
            self.particles.append({'pos': [x, y], 'vel': vel, 'life': life, 'color': color})

    def _get_observation(self):
        self.screen.blit(self.background_surf, (0, 0))
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _create_background(self):
        bg = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        for y in range(self.SCREEN_HEIGHT):
            interp = y / self.SCREEN_HEIGHT
            color = tuple(int(self.COLOR_BG_TOP[i] * (1 - interp) + self.COLOR_BG_BOTTOM[i] * interp) for i in range(3))
            pygame.draw.line(bg, color, (0, y), (self.SCREEN_WIDTH, y))
        return bg

    def _render_game(self):
        self._render_obstacles()
        self._render_particles()
        if not (self.game_over and not self.win):
            self._render_player()

    def _render_player(self):
        rect = self._get_player_rect()
        for i in range(10, 0, -2):
            glow_alpha = 50 - i * 5
            s = pygame.Surface((rect.width + i * 2, rect.height + i * 2), pygame.SRCALPHA)
            pygame.draw.rect(s, (*self.COLOR_PLAYER, glow_alpha), s.get_rect(), border_radius=6)
            self.screen.blit(s, (rect.x - i, rect.y - i))
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, rect, border_radius=4)
        pygame.draw.rect(self.screen, (200, 255, 220), rect.inflate(-6, -6), border_radius=3)

    def _get_player_rect(self):
        return pygame.Rect(self.player_screen_pos[0] - self.PLAYER_SIZE / 2, self.player_screen_pos[1] - self.PLAYER_SIZE / 2, self.PLAYER_SIZE, self.PLAYER_SIZE)

    def _render_obstacles(self):
        for obs in self.obstacles:
            screen_rect = obs['rect'].copy()
            screen_rect.y -= self.camera_y
            obs['screen_rect'] = screen_rect

            if screen_rect.bottom < 0 or screen_rect.top > self.SCREEN_HEIGHT: continue
            
            color = obs['color']
            if obs['type'] == 'rect': pygame.draw.rect(self.screen, color, screen_rect, border_radius=3)
            elif obs['type'] == 'circle':
                pygame.gfxdraw.aacircle(self.screen, int(screen_rect.centerx), int(screen_rect.centery), int(screen_rect.width / 2), color)
                pygame.gfxdraw.filled_circle(self.screen, int(screen_rect.centerx), int(screen_rect.centery), int(screen_rect.width / 2), color)
            elif obs['type'] == 'triangle':
                points = [(screen_rect.midtop), (screen_rect.bottomleft), (screen_rect.bottomright)]
                pygame.gfxdraw.aapolygon(self.screen, points, color)
                pygame.gfxdraw.filled_polygon(self.screen, points, color)

    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, min(255, int(p['life'] * 8)))
            s = pygame.Surface((4,4), pygame.SRCALPHA)
            pygame.draw.circle(s, (*p['color'], alpha), (2,2), 2)
            self.screen.blit(s, (p['pos'][0] - 2, p['pos'][1] - 2))

    def _render_ui(self):
        score_text = self.font_ui.render(f"SCORE: {int(self.score):06d}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 10, 10))
        level_text = self.font_ui.render(f"LEVEL: {self.level}", True, self.COLOR_UI_TEXT)
        self.screen.blit(level_text, (10, 10))

        if self.game_over:
            msg = "ASCENDED!" if self.win else "GAME OVER"
            color = self.COLOR_PLAYER if self.win else self.COLOR_OBSTACLE_TRI
            end_text = self.font_game_over.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            
            shadow_text = self.font_game_over.render(msg, True, (0,0,0))
            self.screen.blit(shadow_text, text_rect.move(2,2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "level": self.level}
        
    def close(self):
        pygame.quit()
        super().close()

    def validate_implementation(self):
        """Call this to verify the environment's implementation against the brief."""
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")