import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Short, user-facing control string
    user_guide = (
        "Controls: Press Space to jump over obstacles in time with the beat."
    )

    # Short, user-facing description of the game
    game_description = (
        "A side-scrolling rhythm game. Jump over procedurally generated obstacles in sync with the music to score points."
    )

    # Frames auto-advance for real-time gameplay
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    MAX_STEPS = 1000

    # Colors
    COLOR_BG_TOP = (20, 0, 40)
    COLOR_BG_BOTTOM = (40, 0, 80)
    COLOR_GROUND = (150, 150, 170)
    COLOR_PLAYER = (0, 200, 255)
    COLOR_OBSTACLE = (255, 50, 50)
    COLOR_PULSE = (50, 255, 50)
    COLOR_UI = (255, 255, 255)

    # Player Physics
    PLAYER_SIZE = 24
    PLAYER_X_POS = SCREEN_WIDTH // 5
    GRAVITY = 0.8
    JUMP_STRENGTH = -14
    GROUND_Y = SCREEN_HEIGHT - 50

    # Rhythm
    BPM = 120
    FRAMES_PER_BEAT = (FPS * 60) / BPM

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font = pygame.font.Font(pygame.font.get_default_font(), 24)
            self.small_font = pygame.font.Font(pygame.font.get_default_font(), 16)
        except IOError:
            self.font = pygame.font.SysFont("sans", 24)
            self.small_font = pygame.font.SysFont("sans", 16)

        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.lives = 0
        self.player_pos = [0, 0]
        self.player_vel_y = 0
        self.player_on_ground = True
        self.player_rect = pygame.Rect(0, 0, 0, 0)
        self.obstacles = []
        self.cleared_obstacles = set()
        self.obstacle_speed = 0
        self.beat_timer = 0
        self.next_beat_frame = 0
        self.last_space_held = False
        
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.lives = 3
        
        # Player state
        self.player_pos = [self.PLAYER_X_POS, self.GROUND_Y]
        self.player_vel_y = 0
        self.player_on_ground = True
        self.player_rect = pygame.Rect(0, 0, self.PLAYER_SIZE, self.PLAYER_SIZE)
        self.player_rect.centerx = self.PLAYER_X_POS
        self.player_rect.bottom = self.GROUND_Y

        # Game state
        self.obstacles = []
        self.cleared_obstacles = set()
        self.obstacle_speed = 5.0 # pixels per frame
        self.last_space_held = False
        
        # Rhythm state
        self.beat_timer = 0
        self.next_beat_frame = self.FRAMES_PER_BEAT
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0.0
        
        # --- Update Game Logic ---
        # First, apply physics from the previous state.
        self._update_player()
        self._update_rhythm_and_obstacles()

        # --- Process new action ---
        # Unpack action
        space_held = action[1] == 1
        
        # Handle jump input on rising edge. The effect (setting velocity) will be applied
        # in the next step's _update_player call.
        if space_held and not self.last_space_held and self.player_on_ground:
            self.player_vel_y = self.JUMP_STRENGTH
            self.player_on_ground = False
            # SFX: Jump sound
        self.last_space_held = space_held
        
        # --- Collision and Scoring ---
        collision_reward = self._handle_collisions()
        clear_reward = self._handle_obstacle_clearing()
        reward += collision_reward + clear_reward

        # --- Step-based rewards and progression ---
        self.steps += 1
        reward += 0.1  # Survival reward for each step

        # Difficulty scaling
        if self.steps > 0 and self.steps % 200 == 0:
            self.obstacle_speed += 0.5 # Increase speed
        
        # --- Check Termination ---
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        if terminated and not self.game_over: # Reached the end
            reward += 50.0 # Victory reward
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _update_player(self):
        # Apply gravity
        self.player_vel_y += self.GRAVITY
        self.player_pos[1] += self.player_vel_y
        
        # Check for ground collision
        if self.player_pos[1] >= self.GROUND_Y:
            self.player_pos[1] = self.GROUND_Y
            self.player_vel_y = 0
            self.player_on_ground = True
        
        self.player_rect.bottom = int(self.player_pos[1])

    def _update_rhythm_and_obstacles(self):
        self.beat_timer += 1
        
        # Spawn new obstacles on the beat
        if self.beat_timer >= self.next_beat_frame:
            self.next_beat_frame += self.FRAMES_PER_BEAT
            self._spawn_obstacle()
        
        # Move existing obstacles and remove off-screen ones
        new_obstacles = []
        for obs in self.obstacles:
            obs['rect'].x -= self.obstacle_speed
            if obs['rect'].right > 0:
                new_obstacles.append(obs)
        self.obstacles = new_obstacles

    def _spawn_obstacle(self):
        if self.np_random.random() < 0.75: # 75% chance to spawn on a beat
            height = self.np_random.integers(20, 51)
            width = self.np_random.integers(20, 41)
            rect = pygame.Rect(
                self.SCREEN_WIDTH,
                self.GROUND_Y - height,
                width,
                height
            )
            obstacle_id = self.steps + rect.x 
            self.obstacles.append({'rect': rect, 'id': obstacle_id})

    def _handle_collisions(self):
        for obs in self.obstacles:
            if self.player_rect.colliderect(obs['rect']):
                self.lives -= 1
                self.game_over = self.lives <= 0
                self.obstacles = [] # Clear screen for recovery
                # SFX: Collision/damage sound
                return -10.0
        return 0.0
        
    def _handle_obstacle_clearing(self):
        reward = 0.0
        for obs in self.obstacles:
            if obs['rect'].right < self.player_rect.left and obs['id'] not in self.cleared_obstacles:
                self.cleared_obstacles.add(obs['id'])
                self.score += 1
                reward += 1.0
                # SFX: Point score sound
        return reward

    def _get_observation(self):
        self._render_background()
        self._render_game_elements()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        # Draw gradient background
        top_color = self.COLOR_BG_TOP
        bottom_color = self.COLOR_BG_BOTTOM
        for y in range(self.SCREEN_HEIGHT):
            interp = y / self.SCREEN_HEIGHT
            r = int(top_color[0] * (1 - interp) + bottom_color[0] * interp)
            g = int(top_color[1] * (1 - interp) + bottom_color[1] * interp)
            b = int(top_color[2] * (1 - interp) + bottom_color[2] * interp)
            pygame.draw.line(self.screen, (r, g, b), (0, y), (self.SCREEN_WIDTH, y))

        # Render rhythm visualizers
        beat_progress = (self.beat_timer % self.FRAMES_PER_BEAT) / self.FRAMES_PER_BEAT
        pulse = math.sin(beat_progress * math.pi)
        
        # Background pulsating circles
        for i in range(3):
            radius = int(50 + pulse * 100 + i * 150)
            alpha = int(30 * (1 - pulse) * (1 - i / 3))
            if radius > 0 and alpha > 0:
                pygame.gfxdraw.aacircle(self.screen, self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2, radius, (*self.COLOR_PULSE, alpha))

    def _render_game_elements(self):
        # Draw ground
        pygame.draw.line(self.screen, self.COLOR_GROUND, (0, self.GROUND_Y), (self.SCREEN_WIDTH, self.GROUND_Y), 3)

        # Draw obstacles
        for obs in self.obstacles:
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, obs['rect'])
            # Add a slight inner highlight for depth
            highlight_rect = obs['rect'].inflate(-4, -4)
            highlight_color = tuple(min(255, c + 50) for c in self.COLOR_OBSTACLE)
            pygame.draw.rect(self.screen, highlight_color, highlight_rect, 2)

        # Draw player with glow
        glow_radius = int(self.PLAYER_SIZE * 0.8)
        glow_center = self.player_rect.center
        for i in range(glow_radius, 0, -2):
            alpha = int(60 * (1 - i / glow_radius)**2)
            pygame.gfxdraw.filled_circle(self.screen, glow_center[0], glow_center[1], i, (*self.COLOR_PLAYER, alpha))
        
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, self.player_rect)

        # Draw finish line near the end
        if self.steps > self.MAX_STEPS - 200:
            progress_to_end = self.steps - (self.MAX_STEPS - 200)
            finish_x = self.SCREEN_WIDTH - (progress_to_end * self.obstacle_speed)
            if finish_x < self.SCREEN_WIDTH * 1.5:
                for y_offset in range(0, self.GROUND_Y, 20):
                    color = (255, 255, 255) if (y_offset // 20) % 2 == 0 else (100, 100, 100)
                    pygame.draw.rect(self.screen, color, (int(finish_x), y_offset, 10, 10))


    def _render_ui(self):
        # Render score
        score_text = self.font.render(f"SCORE: {self.score}", True, self.COLOR_UI)
        self.screen.blit(score_text, (10, 10))

        # Render lives
        lives_text = self.small_font.render("LIVES:", True, self.COLOR_UI)
        self.screen.blit(lives_text, (self.SCREEN_WIDTH - 120, 15))
        for i in range(self.lives):
            life_rect = pygame.Rect(self.SCREEN_WIDTH - 70 + (i * (self.PLAYER_SIZE*0.7 + 5)), 15, int(self.PLAYER_SIZE*0.7), int(self.PLAYER_SIZE*0.7))
            pygame.draw.rect(self.screen, self.COLOR_PLAYER, life_rect)
            
        # Render progress bar
        progress = self.steps / self.MAX_STEPS
        bar_width = self.SCREEN_WIDTH - 20
        bar_height = 5
        pygame.draw.rect(self.screen, (50, 50, 50), (10, self.SCREEN_HEIGHT - 15, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_PULSE, (10, self.SCREEN_HEIGHT - 15, bar_width * progress, bar_height))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
        }

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space
        # Need to reset to initialize pygame surfaces properly
        self.reset(seed=0)
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset(seed=0)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        # Test specific mechanics
        self.reset(seed=0)
        initial_speed = self.obstacle_speed
        for _ in range(201):
            self.step(self.action_space.sample())
        assert self.obstacle_speed > initial_speed, "Speed should increase after 200 steps"
        
        self.reset(seed=0)
        self.player_pos[1] = self.GROUND_Y
        self.player_on_ground = True
        self.last_space_held = False
        self.step([0, 1, 0]) # Jump action
        assert self.player_vel_y == self.JUMP_STRENGTH, "Jump action should set vertical velocity"

        # print("✓ Implementation validated successfully")