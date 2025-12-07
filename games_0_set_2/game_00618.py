
# Generated: 2025-08-27T14:14:32.730820
# Source Brief: brief_00618.md
# Brief Index: 618

        
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
        "Controls: Use arrow keys to move your square. Dodge the red circles."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A minimalist arcade survival game. Dodge an ever-increasing swarm of circles for 30 seconds to win."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 60
    MAX_STEPS = 30 * FPS  # 30 seconds to win

    COLOR_BG = (20, 25, 40)
    COLOR_PLAYER = (255, 255, 255)
    COLOR_ENEMY = (255, 80, 80)
    COLOR_TEXT = (240, 240, 240)
    COLOR_OVERLAY = (20, 25, 40, 180) # Semi-transparent overlay

    PLAYER_SIZE = 20
    PLAYER_SPEED = 4

    BASE_ENEMY_SPEED = 1.5
    BASE_ENEMY_RADIUS = 8
    ENEMY_RADIUS_SCALE = 5 # How much radius increases per unit of speed

    DIFFICULTY_INTERVAL = 100 # Increase difficulty every 100 steps
    SPEED_INCREMENT = 0.05
    SPAWN_CHANCE_BASE = 1.0 / FPS # 1 per second on average
    SPAWN_CHANCE_INCREMENT_FACTOR = 1.01 # 1% increase

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
        self.font_main = pygame.font.SysFont("consolas", 24, bold=True)
        self.font_large = pygame.font.SysFont("consolas", 72, bold=True)
        
        # Initialize state variables
        self.player_pos = None
        self.enemies = []
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.win = False
        self.current_enemy_speed = self.BASE_ENEMY_SPEED
        self.current_spawn_chance = self.SPAWN_CHANCE_BASE
        
        # Initialize state variables
        self.reset()
        # self.validate_implementation() # Optional: for debugging during development
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.win = False

        self.player_pos = pygame.Vector2(
            self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2
        )
        self.enemies = []
        
        # Reset difficulty
        self.current_enemy_speed = self.BASE_ENEMY_SPEED
        self.current_spawn_chance = self.SPAWN_CHANCE_BASE

        # Spawn a few initial enemies to start the action
        for _ in range(3):
            self._spawn_enemy()
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            reward = 0.0
            terminated = True
            return (
                self._get_observation(),
                reward,
                terminated,
                False,
                self._get_info(),
            )

        self.steps += 1
        
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        # Update game logic
        self._update_player(movement)
        self._update_enemies()
        self._handle_spawning_and_difficulty()
        
        # Check for collisions
        player_rect = pygame.Rect(self.player_pos.x, self.player_pos.y, self.PLAYER_SIZE, self.PLAYER_SIZE)
        if self._check_collisions(player_rect):
            self.game_over = True
            # sfx: player_death_sound

        terminated = self.game_over
        reward = 0.0

        # Check for win condition
        if not terminated and self.steps >= self.MAX_STEPS:
            self.win = True
            self.game_over = True
            terminated = True
            reward += 100.0 # Win bonus
            # sfx: win_jingle
        
        # Calculate reward
        reward += self._calculate_reward(movement)
        self.score += reward

        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _calculate_reward(self, movement):
        # Continuous feedback rewards: +0.1 per frame survived.
        # -0.2 if no movement is made (action 0) for a frame, discouraging inaction.
        r = 0.1
        if movement == 0:
            r -= 0.2
        return r

    def _update_player(self, movement):
        # 0=none, 1=up, 2=down, 3=left, 4=right
        if movement == 1:
            self.player_pos.y -= self.PLAYER_SPEED
        elif movement == 2:
            self.player_pos.y += self.PLAYER_SPEED
        elif movement == 3:
            self.player_pos.x -= self.PLAYER_SPEED
        elif movement == 4:
            self.player_pos.x += self.PLAYER_SPEED
        
        # Clamp player position to screen bounds
        self.player_pos.x = np.clip(self.player_pos.x, 0, self.SCREEN_WIDTH - self.PLAYER_SIZE)
        self.player_pos.y = np.clip(self.player_pos.y, 0, self.SCREEN_HEIGHT - self.PLAYER_SIZE)

    def _update_enemies(self):
        for enemy in self.enemies:
            enemy["pos"] += enemy["vel"]
            
            # Bounce off walls
            if enemy["pos"].x - enemy["radius"] < 0 or enemy["pos"].x + enemy["radius"] > self.SCREEN_WIDTH:
                enemy["vel"].x *= -1
                enemy["pos"].x = np.clip(enemy["pos"].x, enemy["radius"], self.SCREEN_WIDTH - enemy["radius"])
                # sfx: bounce_sound
            if enemy["pos"].y - enemy["radius"] < 0 or enemy["pos"].y + enemy["radius"] > self.SCREEN_HEIGHT:
                enemy["vel"].y *= -1
                enemy["pos"].y = np.clip(enemy["pos"].y, enemy["radius"], self.SCREEN_HEIGHT - enemy["radius"])
                # sfx: bounce_sound

    def _handle_spawning_and_difficulty(self):
        # Increase difficulty over time
        if self.steps > 0 and self.steps % self.DIFFICULTY_INTERVAL == 0:
            self.current_enemy_speed += self.SPEED_INCREMENT
            self.current_spawn_chance *= self.SPAWN_CHANCE_INCREMENT_FACTOR
            # sfx: difficulty_up_chime

        # Spawn new enemies based on chance
        if self.np_random.random() < self.current_spawn_chance:
            self._spawn_enemy()

    def _spawn_enemy(self):
        # Spawn on a random edge
        edge = self.np_random.integers(4)
        if edge == 0: # Top
            pos = pygame.Vector2(self.np_random.uniform(0, self.SCREEN_WIDTH), -20)
        elif edge == 1: # Bottom
            pos = pygame.Vector2(self.np_random.uniform(0, self.SCREEN_WIDTH), self.SCREEN_HEIGHT + 20)
        elif edge == 2: # Left
            pos = pygame.Vector2(-20, self.np_random.uniform(0, self.SCREEN_HEIGHT))
        else: # Right
            pos = pygame.Vector2(self.SCREEN_WIDTH + 20, self.np_random.uniform(0, self.SCREEN_HEIGHT))

        # Aim towards the center of the screen with some randomness
        target = pygame.Vector2(
            self.SCREEN_WIDTH / 2 + self.np_random.uniform(-100, 100),
            self.SCREEN_HEIGHT / 2 + self.np_random.uniform(-100, 100)
        )
        vel = (target - pos).normalize() * self.current_enemy_speed
        
        radius = self.BASE_ENEMY_RADIUS + (self.current_enemy_speed - self.BASE_ENEMY_SPEED) * self.ENEMY_RADIUS_SCALE

        self.enemies.append({"pos": pos, "vel": vel, "radius": max(4, radius)})
        # sfx: enemy_spawn_whoosh

    def _check_collisions(self, player_rect):
        for enemy in self.enemies:
            # Find the closest point on the player rectangle to the circle's center
            closest_x = max(player_rect.left, min(enemy["pos"].x, player_rect.right))
            closest_y = max(player_rect.top, min(enemy["pos"].y, player_rect.bottom))
            
            distance_sq = (closest_x - enemy["pos"].x)**2 + (closest_y - enemy["pos"].y)**2
            
            if distance_sq < (enemy["radius"]**2):
                return True
        return False

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
        # Render enemies
        for enemy in self.enemies:
            pos_int = (int(enemy["pos"].x), int(enemy["pos"].y))
            radius_int = int(enemy["radius"])
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], radius_int, self.COLOR_ENEMY)
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], radius_int, self.COLOR_ENEMY)

        # Render player
        player_rect = pygame.Rect(self.player_pos.x, self.player_pos.y, self.PLAYER_SIZE, self.PLAYER_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect)
        
    def _render_ui(self):
        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        time_text = self.font_main.render(f"TIME: {time_left:.2f}", True, self.COLOR_TEXT)
        self.screen.blit(time_text, (10, 10))

        score_text = self.font_main.render(f"SCORE: {self.score:.1f}", True, self.COLOR_TEXT)
        score_text_rect = score_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(score_text, score_text_rect)
        
        # Render game over/win message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill(self.COLOR_OVERLAY)
            self.screen.blit(overlay, (0, 0))

            if self.win:
                msg_text = self.font_large.render("YOU WIN!", True, self.COLOR_PLAYER)
            else:
                msg_text = self.font_large.render("GAME OVER", True, self.COLOR_ENEMY)
            
            msg_rect = msg_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(msg_text, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "win": self.win,
            "enemy_count": len(self.enemies),
            "enemy_speed": self.current_enemy_speed,
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