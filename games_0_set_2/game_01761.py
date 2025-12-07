import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A procedurally generated side-scrolling platformer where an agent must navigate
    treacherous gaps and reach the end goal within a time limit. The game features
    simple geometric shapes, vibrant retro colors, and physics-based movement.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Short, user-facing control string
    user_guide = "Controls: ←→ to move, ↑ to jump."

    # Short, user-facing description of the game
    game_description = (
        "A retro side-scrolling platformer. Navigate across platforms, avoid the moving red orbs, "
        "and reach the flag at the end of each stage before time runs out!"
    )

    # Frames auto-advance at a fixed rate
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.GRAVITY = 0.7
        self.JUMP_STRENGTH = -14
        self.PLAYER_SPEED = 5
        self.PLAYER_SIZE = 20
        self.MAX_STAGES = 3
        self.TIME_PER_STAGE = 40  # in seconds

        # --- Colors ---
        self.COLOR_BG_TOP = (50, 0, 80)  # Dark Purple
        self.COLOR_BG_BOTTOM = (0, 80, 100) # Dark Teal
        self.COLOR_PLAYER = (0, 150, 255)  # Bright Blue
        self.COLOR_PLAYER_FLASH = (255, 255, 255) # White
        self.COLOR_PLATFORM = (0, 200, 100) # Bright Green
        self.COLOR_OBSTACLE = (255, 50, 50) # Bright Red
        self.COLOR_FLAG = (255, 220, 0) # Bright Yellow
        self.COLOR_FLAG_POLE = (200, 200, 200) # Light Grey
        self.COLOR_TEXT = (255, 255, 255) # White

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        self._background_surface = self._create_background()

        # --- Game State Variables (initialized in reset) ---
        self.player_pos = None
        self.player_vel = None
        self.on_ground = False
        self.platforms = []
        self.obstacles = []
        self.end_flag = None
        self.player_start_pos = [0, 0]

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.current_stage = 1
        self.time_remaining = 0
        self.obstacle_speed_modifier = 0.0
        self.damage_flash_timer = 0

        # --- Initialize and Validate ---
        # self.reset() is called here, but we need to initialize state variables first
        # to avoid errors if reset() is called by a wrapper before __init__ completes.
        # The first call to reset() will properly set up the game.

    def _create_background(self):
        """Pre-renders the background gradient for performance."""
        bg = pygame.Surface((self.WIDTH, self.HEIGHT))
        for y in range(self.HEIGHT):
            interp = y / self.HEIGHT
            color = (
                int(self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp),
                int(self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp),
                int(self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp),
            )
            pygame.draw.line(bg, color, (0, y), (self.WIDTH, y))
        return bg

    def _generate_level(self, stage):
        """Creates the layout for a given stage."""
        self.platforms.clear()
        self.obstacles.clear()

        # Ground floor
        self.platforms.append(pygame.Rect(0, self.HEIGHT - 20, self.WIDTH, 20))

        if stage == 1:
            self.player_start_pos = [50, self.HEIGHT - 40]
            self.platforms.extend([
                pygame.Rect(150, self.HEIGHT - 80, 100, 20),
                pygame.Rect(300, self.HEIGHT - 140, 100, 20),
                pygame.Rect(450, self.HEIGHT - 200, 120, 20),
            ])
            self.obstacles.append({
                'base_pos': [350, self.HEIGHT - 80], 'size': 20,
                'amplitude': 40, 'frequency': 0.03, 'offset': 0
            })
            self.end_flag = pygame.Rect(590, self.HEIGHT - 250, 30, 50)
        elif stage == 2:
            self.player_start_pos = [30, self.HEIGHT - 40]
            self.platforms.extend([
                pygame.Rect(120, self.HEIGHT - 60, 80, 20),
                pygame.Rect(250, self.HEIGHT - 40, 60, 20),
                pygame.Rect(360, self.HEIGHT - 120, 80, 20),
                pygame.Rect(250, self.HEIGHT - 200, 60, 20),
                pygame.Rect(400, self.HEIGHT - 280, 150, 20),
            ])
            self.obstacles.extend([
                {'base_pos': [200, self.HEIGHT - 150], 'size': 20, 'amplitude': 80, 'frequency': 0.04, 'offset': 0},
                {'base_pos': [480, self.HEIGHT - 180], 'size': 15, 'amplitude': 100, 'frequency': -0.05, 'offset': 1.5}
            ])
            self.end_flag = pygame.Rect(560, self.HEIGHT - 330, 30, 50)
        elif stage == 3:
            self.player_start_pos = [30, self.HEIGHT - 40]
            self.platforms.extend([
                pygame.Rect(100, self.HEIGHT - 100, 50, 20),
                pygame.Rect(220, self.HEIGHT - 150, 50, 20),
                pygame.Rect(340, self.HEIGHT - 200, 50, 20),
                pygame.Rect(220, self.HEIGHT - 280, 50, 20),
                pygame.Rect(450, self.HEIGHT - 320, 100, 20),
            ])
            self.obstacles.extend([
                {'base_pos': [160, self.HEIGHT - 100], 'size': 20, 'amplitude': 50, 'frequency': 0.08, 'offset': 0},
                {'base_pos': [280, self.HEIGHT - 200], 'size': 20, 'amplitude': 80, 'frequency': -0.08, 'offset': 0.5},
                {'base_pos': [400, self.HEIGHT - 150], 'size': 25, 'amplitude': 120, 'frequency': 0.06, 'offset': 1.0}
            ])
            self.end_flag = pygame.Rect(580, self.HEIGHT - 370, 30, 50)

        # Reset player to the start of the newly generated stage
        self.player_pos = list(self.player_start_pos)
        self.player_vel = [0, 0]
        self.on_ground = False

        # FIX: Initialize obstacle positions after they are created.
        self._update_obstacles()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.current_stage = 1
        self.time_remaining = self.TIME_PER_STAGE * self.FPS
        self.obstacle_speed_modifier = 1.0
        self.damage_flash_timer = 0

        self._generate_level(1)

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Action Handling ---
        movement = action[0]  # 0=none, 1=up, 2=down, 3=left, 4=right
        
        # --- Game Logic Update ---
        self._update_player(movement)
        self._update_obstacles()
        
        # --- Collision, Events, and Rewards ---
        event_reward = self._handle_events()
        
        # --- State and Timers Update ---
        self.time_remaining -= 1
        self.steps += 1
        if self.damage_flash_timer > 0:
            self.damage_flash_timer -= 1

        # --- Reward Calculation ---
        # +0.01 per frame alive (scaled down to keep rewards in a reasonable range)
        reward = 0.01 + event_reward
        self.score += reward

        # --- Termination Check ---
        if self.time_remaining <= 0:
            self.game_over = True
        
        terminated = self.game_over

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_player(self, movement):
        # Horizontal movement
        if movement == 3:  # Left
            self.player_vel[0] = -self.PLAYER_SPEED
        elif movement == 4:  # Right
            self.player_vel[0] = self.PLAYER_SPEED
        else:
            self.player_vel[0] = 0

        # Jumping
        if movement == 1 and self.on_ground:  # Up
            self.player_vel[1] = self.JUMP_STRENGTH
            self.on_ground = False
            # Sound: Jump sfx

        # Apply gravity
        self.player_vel[1] += self.GRAVITY
        self.player_vel[1] = min(self.player_vel[1], 15)  # Terminal velocity

        # Update position (before collision correction)
        self.player_pos[0] += self.player_vel[0]
        self.player_pos[1] += self.player_vel[1]

        # Keep player within horizontal screen bounds
        self.player_pos[0] = max(0, min(self.player_pos[0], self.WIDTH - self.PLAYER_SIZE))
    
    def _update_obstacles(self):
        """Update obstacle positions based on sinusoidal movement."""
        for obs in self.obstacles:
            t = self.steps * obs['frequency'] * self.obstacle_speed_modifier + obs['offset']
            obs['pos'] = [
                obs['base_pos'][0],
                obs['base_pos'][1] + math.sin(t) * obs['amplitude']
            ]

    def _handle_events(self):
        """Handles all collisions and game events, returning event-based rewards."""
        reward = 0
        player_rect_prev_y = pygame.Rect(self.player_pos[0] - self.player_vel[0], self.player_pos[1] - self.player_vel[1], self.PLAYER_SIZE, self.PLAYER_SIZE)
        player_rect = pygame.Rect(self.player_pos[0], self.player_pos[1], self.PLAYER_SIZE, self.PLAYER_SIZE)
        
        # 1. Fall off screen (Termination)
        if player_rect.top > self.HEIGHT:
            self.game_over = True
            return -1 # Small penalty for falling

        # 2. Platform collision (Landing)
        self.on_ground = False
        for plat in self.platforms:
            if player_rect.colliderect(plat) and self.player_vel[1] >= 0:
                # Check if player was above the platform in the previous frame
                if player_rect_prev_y.bottom <= plat.top:
                    self.player_pos[1] = plat.top - self.PLAYER_SIZE
                    self.player_vel[1] = 0
                    self.on_ground = True
                    break # Land on the first platform found

        # Refresh player rect after y-correction
        player_rect = pygame.Rect(self.player_pos[0], self.player_pos[1], self.PLAYER_SIZE, self.PLAYER_SIZE)

        # 3. Obstacle collision (Penalty + Reset Stage)
        for obs in self.obstacles:
            obs_rect = pygame.Rect(obs['pos'][0], obs['pos'][1], obs['size'], obs['size'])
            if player_rect.colliderect(obs_rect):
                reward -= 5
                self.damage_flash_timer = 10  # Flash for 10 frames
                self._generate_level(self.current_stage) # Reset to start of current stage
                # Sound: Damage sfx
                return reward  # Return immediately after hit

        # 4. End flag collision (Progress Stage / Win)
        if player_rect.colliderect(self.end_flag):
            if self.current_stage == self.MAX_STAGES:
                reward += 50  # Big reward for winning
                self.game_over = True
                # Sound: Final Win sfx
            else:
                reward += 10  # Reward for clearing a stage
                self.current_stage += 1
                self.obstacle_speed_modifier += 0.05
                self.time_remaining = self.TIME_PER_STAGE * self.FPS # Reset timer
                self._generate_level(self.current_stage)
                # Sound: Stage Clear sfx
        
        return reward

    def _get_observation(self):
        # Clear screen with pre-rendered background
        self.screen.blit(self._background_surface, (0, 0))
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw platforms
        for plat in self.platforms:
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM, plat)

        # Draw obstacles
        for obs in self.obstacles:
            pygame.draw.circle(self.screen, self.COLOR_OBSTACLE, (int(obs['pos'][0] + obs['size']/2), int(obs['pos'][1] + obs['size']/2)), obs['size'] / 2)

        # Draw end flag
        pole_rect = pygame.Rect(self.end_flag.x, self.end_flag.y, 5, self.end_flag.height)
        pygame.draw.rect(self.screen, self.COLOR_FLAG_POLE, pole_rect)
        flag_points = [
            (self.end_flag.x + 5, self.end_flag.y),
            (self.end_flag.x + 25, self.end_flag.y + 10),
            (self.end_flag.x + 5, self.end_flag.y + 20)
        ]
        pygame.draw.polygon(self.screen, self.COLOR_FLAG, flag_points)

        # Draw player
        player_color = self.COLOR_PLAYER_FLASH if self.damage_flash_timer > 0 else self.COLOR_PLAYER
        player_rect = pygame.Rect(int(self.player_pos[0]), int(self.player_pos[1]), self.PLAYER_SIZE, self.PLAYER_SIZE)
        pygame.draw.rect(self.screen, player_color, player_rect)
        # Add a border for better visibility
        pygame.draw.rect(self.screen, (255,255,255), player_rect, 1)

    def _render_ui(self):
        # Timer
        time_str = f"Time: {max(0, self.time_remaining // self.FPS)}"
        time_text = self.font.render(time_str, True, self.COLOR_TEXT)
        self.screen.blit(time_text, (10, 10))

        # Stage
        stage_str = f"Stage: {self.current_stage}/{self.MAX_STAGES}"
        stage_text = self.font.render(stage_str, True, self.COLOR_TEXT)
        stage_rect = stage_text.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(stage_text, stage_rect)
        
        # Score
        score_str = f"Score: {int(self.score)}"
        score_text = self.small_font.render(score_str, True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 45))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stage": self.current_stage,
            "time_remaining": self.time_remaining,
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()