import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move. Hold Space near a toxic spill to clean it up."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Race against time to clean up 10 toxic spills in a haunted, contaminated facility. "
        "The contamination spreads from spills, slowing you down. Work fast before time runs out!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and rendering constants
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.FPS = 30

        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.FONT = pygame.font.SysFont("Consolas", 20, bold=True)
        self.BIG_FONT = pygame.font.SysFont("Consolas", 50, bold=True)
        
        # Colors
        self.COLOR_BG = (25, 35, 25)
        self.COLOR_PLAYER = (255, 255, 255)
        self.COLOR_SPILL = (220, 220, 0)
        self.COLOR_CONTAMINATION = (10, 20, 10)
        self.COLOR_UI = (200, 255, 200)
        self.COLOR_CLEAN_EFFECT = (150, 255, 150)
        
        # Game constants
        self.PLAYER_SIZE = 10
        self.PLAYER_SPEED = 4.0
        self.PLAYER_SLOW_SPEED = 2.0
        self.SPILL_BASE_RADIUS = 15
        self.CLEANING_RADIUS = 40
        self.MAX_SPILLS = 10
        self.GAME_DURATION = 90.0 # seconds

        # State variables (initialized in reset)
        self.player_pos = None
        self.spills = None
        self.score = None
        self.timer = None
        self.steps = None
        self.game_over = None
        self.base_contamination_spread_rate = None
        self.current_spread_rate = None
        self.cleaning_effects = None
        self.last_space_state = False
        self.last_contamination_penalty = 0

        # Surfaces for rendering
        self.contamination_surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        self._background_texture = None

        # Initialize state
        self.reset()
        # self.validate_implementation() # This can be run by the user if needed
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Create background texture on first reset, ensuring np_random is available
        if self._background_texture is None:
            self._background_texture = self._create_background_texture()

        self.player_pos = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2], dtype=np.float32)
        self.spills = []
        self.cleaning_effects = []
        self.score = 0
        self.timer = self.GAME_DURATION
        self.steps = 0
        self.game_over = False
        self.base_contamination_spread_rate = 0.1 # pixels per frame
        self._update_spread_rate()
        self.last_space_state = False
        self.last_contamination_penalty = 0

        self.contamination_surface.fill((0, 0, 0, 0))
        for _ in range(3):
            self._spawn_spill()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            # If the game is over, we still need to return a valid observation
            return self._get_observation(), 0, True, False, self._get_info()

        self.clock.tick(self.FPS)
        
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # --- Update Game Logic ---
        self.steps += 1
        self.timer = max(0, self.timer - 1.0 / self.FPS)
        
        self._update_contamination()
        self._update_player(movement)
        
        reward, cleaned_this_step = self._handle_cleaning(space_held)

        if cleaned_this_step:
            self.score += 1
            self._update_spread_rate()
            if self.score < self.MAX_SPILLS:
                self._spawn_spill()
        
        self._update_effects()

        # Continuous/Periodic Rewards
        if self._is_on_contamination(self.player_pos):
            reward -= 0.01

        # Calculate contamination penalty every second
        if self.steps % self.FPS == 0:
            contamination_penalty = self._calculate_contamination_penalty()
            reward += contamination_penalty
            self.last_contamination_penalty = contamination_penalty
        else:
            reward += self.last_contamination_penalty / self.FPS # Distribute penalty over the second

        # --- Check Termination ---
        terminated = self.timer <= 0 or self.score >= self.MAX_SPILLS
        if terminated and not self.game_over:
            self.game_over = True
            if self.score >= self.MAX_SPILLS:
                reward += 100  # Win bonus
            if self.timer <= 0:
                reward -= 100  # Loss penalty
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_player(self, movement):
        speed = self.PLAYER_SLOW_SPEED if self._is_on_contamination(self.player_pos) else self.PLAYER_SPEED
        
        if movement == 1: # Up
            self.player_pos[1] -= speed
        elif movement == 2: # Down
            self.player_pos[1] += speed
        elif movement == 3: # Left
            self.player_pos[0] -= speed
        elif movement == 4: # Right
            self.player_pos[0] += speed
            
        self.player_pos[0] = np.clip(self.player_pos[0], 0, self.SCREEN_WIDTH - self.PLAYER_SIZE)
        self.player_pos[1] = np.clip(self.player_pos[1], 0, self.SCREEN_HEIGHT - self.PLAYER_SIZE)

    def _handle_cleaning(self, space_held):
        reward = 0
        cleaned_spill = False
        
        # Trigger on key press, not hold
        if space_held and not self.last_space_state:
            # SFX: cleaning_start.wav
            spill_to_remove = None
            for spill in self.spills:
                distance = np.linalg.norm(self.player_pos - spill['pos'])
                if distance < self.CLEANING_RADIUS:
                    spill_to_remove = spill
                    break
            
            if spill_to_remove:
                # SFX: success_chime.wav
                self.spills.remove(spill_to_remove)
                self.cleaning_effects.append({'pos': spill_to_remove['pos'], 'radius': self.SPILL_BASE_RADIUS * 2, 'life': 1.0})
                reward += 10
                cleaned_spill = True

        self.last_space_state = space_held
        return reward, cleaned_spill

    def _update_contamination(self):
        for spill in self.spills:
            spill['contamination_radius'] += self.current_spread_rate
            radius = int(spill['contamination_radius'])
            pos = (int(spill['pos'][0]), int(spill['pos'][1]))
            alpha = min(60, int(radius * 0.5))
            pygame.gfxdraw.filled_circle(self.contamination_surface, pos[0], pos[1], radius, (*self.COLOR_CONTAMINATION, alpha))
    
    def _update_effects(self):
        for effect in self.cleaning_effects[:]:
            effect['life'] -= 0.05
            if effect['life'] <= 0:
                self.cleaning_effects.remove(effect)

    def _update_spread_rate(self):
        # Difficulty scaling: spread rate increases every 2 spills cleaned
        difficulty_tier = self.score // 2
        self.current_spread_rate = self.base_contamination_spread_rate + (difficulty_tier * 0.05)

    def _spawn_spill(self):
        for _ in range(50): # Try 50 times to find a clean spot
            pos = np.array([
                self.np_random.integers(low=30, high=self.SCREEN_WIDTH - 30),
                self.np_random.integers(low=30, high=self.SCREEN_HEIGHT - 30)
            ], dtype=np.float32)
            if not self._is_on_contamination(pos):
                self.spills.append({
                    'pos': pos,
                    'contamination_radius': 0.0,
                    'pulse_offset': self.np_random.random() * 2 * math.pi
                })
                return
        # Fallback: if no clean spot found, just place it anywhere
        pos = np.array([
            self.np_random.integers(low=30, high=self.SCREEN_WIDTH - 30),
            self.np_random.integers(low=30, high=self.SCREEN_HEIGHT - 30)
        ], dtype=np.float32)
        self.spills.append({
            'pos': pos,
            'contamination_radius': 0.0,
            'pulse_offset': self.np_random.random() * 2 * math.pi
        })

    def _is_on_contamination(self, pos):
        if 0 <= pos[0] < self.SCREEN_WIDTH and 0 <= pos[1] < self.SCREEN_HEIGHT:
            return self.contamination_surface.get_at((int(pos[0]), int(pos[1]))).a > 0
        return False

    def _calculate_contamination_penalty(self):
        total_area = self.SCREEN_WIDTH * self.SCREEN_HEIGHT
        contaminated_area = 0
        for spill in self.spills:
            contaminated_area += math.pi * (spill['contamination_radius'] ** 2)
        
        contamination_percent = min(1.0, contaminated_area / total_area)
        penalty = -math.floor(contamination_percent * 10) # -1 for each 10%
        return penalty

    def _get_observation(self):
        self.screen.blit(self._background_texture, (0, 0))
        self.screen.blit(self.contamination_surface, (0, 0))
        
        self._render_spills()
        self._render_player()
        self._render_effects()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_spills(self):
        for spill in self.spills:
            pulse = (math.sin(self.steps * 0.1 + spill['pulse_offset']) + 1) / 2
            radius = int(self.SPILL_BASE_RADIUS + pulse * 5)
            pos = (int(spill['pos'][0]), int(spill['pos'][1]))
            
            # Draw a darker, larger base for depth
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, (150, 150, 0))
            # Draw the bright, pulsating top
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(radius * 0.8), self.COLOR_SPILL)
            # Add an anti-aliased outline for smoothness
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, self.COLOR_SPILL)

    def _render_player(self):
        pos = (int(self.player_pos[0]), int(self.player_pos[1]))
        
        # Glow effect
        glow_radius = int(self.PLAYER_SIZE * 1.5)
        glow_alpha = 100
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], glow_radius, (*self.COLOR_PLAYER, glow_alpha))
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], glow_radius, (*self.COLOR_PLAYER, glow_alpha))
        
        # Main body
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.PLAYER_SIZE, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.PLAYER_SIZE, self.COLOR_PLAYER)

    def _render_effects(self):
        for effect in self.cleaning_effects:
            pos = (int(effect['pos'][0]), int(effect['pos'][1]))
            radius = int(effect['radius'] * effect['life'])
            alpha = int(255 * effect['life'])
            if radius > 0 and alpha > 0:
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, (*self.COLOR_CLEAN_EFFECT, alpha))
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius-1, (*self.COLOR_CLEAN_EFFECT, alpha))

    def _render_ui(self):
        # Score
        score_text = self.FONT.render(f"Spills Cleaned: {self.score}/{self.MAX_SPILLS}", True, self.COLOR_UI)
        self.screen.blit(score_text, (10, 10))
        
        # Timer
        timer_text = self.FONT.render(f"Time: {int(self.timer // 60):02}:{int(self.timer % 60):02}", True, self.COLOR_UI)
        timer_rect = timer_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(timer_text, timer_rect)

        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self.score >= self.MAX_SPILLS:
                msg = "MISSION COMPLETE"
                color = (100, 255, 100)
            else:
                msg = "CONTAINMENT FAILURE"
                color = (255, 100, 100)
            
            end_text = self.BIG_FONT.render(msg, True, color)
            end_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, end_rect)

    def _create_background_texture(self):
        texture = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        texture.fill(self.COLOR_BG)
        for _ in range(2000):
            color = (
                np.clip(self.COLOR_BG[0] + self.np_random.integers(-10, 11), 0, 255),
                np.clip(self.COLOR_BG[1] + self.np_random.integers(-10, 11), 0, 255),
                np.clip(self.COLOR_BG[2] + self.np_random.integers(-10, 11), 0, 255)
            )
            x = self.np_random.integers(0, self.SCREEN_WIDTH + 1)
            y = self.np_random.integers(0, self.SCREEN_HEIGHT + 1)
            w = self.np_random.integers(1, 4)
            h = self.np_random.integers(1, 4)
            pygame.draw.rect(texture, color, (x, y, w, h))
        return texture

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "timer": round(self.timer, 2),
            "spills_on_map": len(self.spills),
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    # pip install gymnasium[classic-control]
    import gymnasium.utils.play

    env = GameEnv()
    
    # Define keys for human player
    # Movement: Arrow keys
    # Action 1 (Space): Space bar
    # Action 2 (Shift): Left Shift
    key_map = {
        (pygame.K_UP,): 1,
        (pygame.K_DOWN,): 2,
        (pygame.K_LEFT,): 3,
        (pygame.K_RIGHT,): 4,
    }
    
    # The `play` utility expects a single value for each key.
    # We will construct the MultiDiscrete action from the pressed keys.
    pressed_keys = []
    
    def get_action(keys):
        # Default action is no-op
        action = [0, 0, 0]
        
        # Movement
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        # Space
        if keys[pygame.K_SPACE]:
            action[1] = 1
            
        # Shift
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1
            
        return action

    # Use gymnasium's play utility
    gymnasium.utils.play.play(env, fps=30, callback=None, keys_to_action=get_action, noop=np.array([0,0,0]))

    env.close()