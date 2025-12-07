import gymnasium as gym
import os
import pygame
import numpy as np
import pygame.gfxdraw
import math
from gymnasium.spaces import MultiDiscrete
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    An idle baseball game where the agent must match the batter's color to the incoming
    ball's color. Successful matches ('strikes') increase the score and contribute to
    persistent upgrades like faster batter switching and unlocking new stadiums. The
    challenge comes from the increasing speed of the pitches over time.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "An idle baseball game where you must match the batter's color to the incoming "
        "ball's color to score points and unlock upgrades."
    )
    user_guide = "Controls: Press space to switch the batter's color to match the incoming ball."
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    MAX_STEPS = 5000
    
    # Colors
    COLOR_BG_STADIUM_1 = (135, 206, 235) # Sky Blue
    COLOR_BG_STADIUM_2 = (255, 165, 0)   # Sunset Orange
    COLOR_BG_STADIUM_3 = (25, 25, 112)   # Midnight Blue
    COLOR_FIELD = (60, 179, 113)         # Medium Sea Green
    COLOR_MOUND = (210, 180, 140)        # Tan
    COLOR_LINES = (255, 255, 255)        # White
    COLOR_UI_TEXT = (255, 255, 255)
    COLOR_UI_SHADOW = (0, 0, 0, 128)

    # Batter/Ball Colors
    COLORS = [
        (255, 69, 58),   # Red
        (48, 209, 88),   # Green
        (10, 132, 255),  # Blue
        (255, 214, 10),  # Yellow
    ]
    COLOR_NAMES = ["RED", "GREEN", "BLUE", "YELLOW"]

    # Game Object Positions
    PITCHER_POS = (580, 260)
    PLATE_POS = (100, 260)
    BATTER_POS = (90, 260)

    # Rewards
    REWARD_STRIKE = 1.0
    REWARD_MISS = -0.1
    REWARD_STADIUM_UNLOCK = 5.0
    REWARD_STRIKE_MILESTONE = 0.1 # Per 10 strikes

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
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
        self.font_main = pygame.font.Font(None, 32)
        self.font_score = pygame.font.Font(None, 48)
        self.font_batter = pygame.font.Font(None, 24)

        # Persistent state (persists across resets)
        self.total_strikes = 0
        self.base_pitch_speed = 3.0
        self.pitch_speed_increase = 0.01
        self.base_switch_cooldown = 30 # frames
        self.switch_cooldown_reduction = 1
        
        self.stadiums = [
            {"name": "Daylight Park", "bg_color": self.COLOR_BG_STADIUM_1, "field_color": self.COLOR_FIELD},
            {"name": "Sunset Field", "bg_color": self.COLOR_BG_STADIUM_2, "field_color": (50, 140, 90)},
            {"name": "Night Arena", "bg_color": self.COLOR_BG_STADIUM_3, "field_color": (40, 110, 70)},
        ]
        self.stadium_unlock_milestones = {1: 100, 2: 500} # index: strikes
        self.unlocked_stadium_indices = [0]
        self.current_stadium_index = 0

        # Initialize all other state variables
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.current_batter_index = 0
        self.ball_pos = [0, 0]
        self.ball_color_index = 0
        self.pitch_in_flight = False
        self.batter_switch_cooldown = 0
        self.batter_switch_cooldown_max = self.base_switch_cooldown
        self.last_space_held = False
        self.particles = []
        
    def _update_progression(self):
        """Checks for and applies persistent upgrades based on total strikes."""
        reward_bonus = 0
        
        # Stadium Unlocks
        for idx, milestone in self.stadium_unlock_milestones.items():
            if self.total_strikes >= milestone and idx not in self.unlocked_stadium_indices:
                self.unlocked_stadium_indices.append(idx)
                self.current_stadium_index = idx
                reward_bonus += self.REWARD_STADIUM_UNLOCK
                # SFX: Fanfare, crowd cheer
        
        # Pitch Speed Increase (every 500 strikes)
        self.pitch_speed = self.base_pitch_speed + (self.total_strikes // 500) * self.pitch_speed_increase
        
        # Batter Switch Cooldown Reduction (every 250 strikes)
        upgrades = self.total_strikes // 250
        self.batter_switch_cooldown_max = max(5, self.base_switch_cooldown - upgrades * self.switch_cooldown_reduction)

        # Strike Milestone Reward
        if self.total_strikes > 0 and self.total_strikes % 10 == 0:
            reward_bonus += self.REWARD_STRIKE_MILESTONE
            
        return reward_bonus

    def _start_new_pitch(self):
        """Resets the ball for a new pitch."""
        self.ball_pos = list(self.PITCHER_POS)
        self.ball_color_index = self.np_random.integers(0, len(self.COLORS))
        self.pitch_in_flight = True
        # SFX: Pitch wind-up

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset episode-specific state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.current_batter_index = 0
        self.batter_switch_cooldown = 0
        self.last_space_held = False
        self.particles = []

        # Update progression based on any strikes from previous episodes
        self._update_progression()
        # Set the first pitch
        self._start_new_pitch()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        
        # Unpack factorized action
        # movement = action[0] # Unused
        space_held = action[1] == 1
        # shift_held = action[2] == 1 # Unused

        # --- Game Logic ---
        self.steps += 1
        
        # 1. Handle Input (Switch Batter)
        space_pressed = space_held and not self.last_space_held
        if space_pressed and self.batter_switch_cooldown == 0:
            self.current_batter_index = (self.current_batter_index + 1) % len(self.COLORS)
            self.batter_switch_cooldown = self.batter_switch_cooldown_max
            # SFX: "Switch" sound, like a UI click
        self.last_space_held = space_held

        # 2. Update Cooldowns
        if self.batter_switch_cooldown > 0:
            self.batter_switch_cooldown -= 1

        # 3. Update Ball
        if self.pitch_in_flight:
            self.ball_pos[0] -= self.pitch_speed
            # Check if ball reached the plate
            if self.ball_pos[0] < self.PLATE_POS[0]:
                self.pitch_in_flight = False
                is_match = self.current_batter_index == self.ball_color_index
                
                if is_match:
                    reward += self.REWARD_STRIKE
                    self.total_strikes += 1
                    self._create_particles(self.PLATE_POS, self.COLORS[self.ball_color_index], 30, 'strike')
                    # SFX: "Crack of the bat"
                    # Check for progression unlocks and add bonus rewards
                    reward += self._update_progression()
                else:
                    reward += self.REWARD_MISS
                    self._create_particles(self.PLATE_POS, (150, 150, 150), 20, 'out')
                    # SFX: "Poof" or "Miss" sound
                
                self._start_new_pitch()

        # 4. Update Particles
        self._update_particles()
        
        self.score += reward
        terminated = self.steps >= self.MAX_STEPS
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _get_observation(self):
        # 1. Clear screen with stadium background
        stadium = self.stadiums[self.current_stadium_index]
        self.screen.fill(stadium["bg_color"])
        
        # 2. Render all game elements
        self._render_field(stadium["field_color"])
        self._render_entities()
        self._render_particles()
        
        # 3. Render UI overlay
        self._render_ui()
        
        # 4. Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "total_strikes": self.total_strikes,
            "current_stadium": self.stadiums[self.current_stadium_index]["name"],
        }

    # --- Rendering Helpers ---

    def _render_field(self, field_color):
        # Field
        pygame.draw.rect(self.screen, field_color, (0, 200, self.WIDTH, self.HEIGHT - 200))
        # Pitcher's Mound
        pygame.gfxdraw.filled_circle(self.screen, self.PITCHER_POS[0], self.PITCHER_POS[1], 20, self.COLOR_MOUND)
        # Batter's Box
        pygame.draw.rect(self.screen, self.COLOR_LINES, (self.PLATE_POS[0] - 30, self.PLATE_POS[1] - 40, 60, 80), 2)

    def _render_entities(self):
        # Ball
        if self.pitch_in_flight:
            ball_color = self.COLORS[self.ball_color_index]
            self._draw_glowing_circle(self.screen, ball_color, self.ball_pos, 10)
        
        # Batter
        batter_color = self.COLORS[self.current_batter_index]
        bx, by = self.BATTER_POS
        # Body
        self._draw_glowing_circle(self.screen, batter_color, (bx, by), 18, transparent=True)
        pygame.gfxdraw.filled_circle(self.screen, int(bx), int(by), 15, batter_color)
        # Head
        self._draw_glowing_circle(self.screen, batter_color, (bx, by-25), 13, transparent=True)
        pygame.gfxdraw.filled_circle(self.screen, int(bx), int(by-25), 10, batter_color)

    def _render_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            alpha = max(0, min(255, int(255 * (p['life'] / p['max_life']))))
            color_with_alpha = p['color'] + (alpha,)
            size = int(p['size'] * (p['life'] / p['max_life']))
            if size > 0:
                # Create a temporary surface for the particle to handle alpha
                particle_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.circle(particle_surf, color_with_alpha, (size, size), size)
                self.screen.blit(particle_surf, (int(p['pos'][0]) - size, int(p['pos'][1]) - size))

    def _render_ui(self):
        # Total Strikes
        self._draw_text(f"STRIKES: {self.total_strikes}", (self.WIDTH - 10, 10), self.font_score, self.COLOR_UI_TEXT, align="topright")
        
        # Batter Status
        batter_color = self.COLORS[self.current_batter_index]
        batter_name = self.COLOR_NAMES[self.current_batter_index]
        self._draw_text(f"BATTER:", (10, 10), self.font_main, self.COLOR_UI_TEXT, align="topleft")
        self._draw_text(batter_name, (120, 10), self.font_main, batter_color, align="topleft")

        # Switch Cooldown
        if self.batter_switch_cooldown > 0:
            cooldown_percent = self.batter_switch_cooldown / self.batter_switch_cooldown_max
            bar_width = 150
            bar_height = 10
            fill_width = int(bar_width * cooldown_percent)
            pygame.draw.rect(self.screen, self.COLOR_UI_SHADOW, (10, 50, bar_width, bar_height))
            pygame.draw.rect(self.screen, self.COLORS[self.current_batter_index], (10, 50, fill_width, bar_height))
            
    # --- Utility Helpers ---

    def _draw_text(self, text, pos, font, color, align="topleft"):
        text_surf = font.render(text, True, color)
        text_shadow = font.render(text, True, self.COLOR_UI_SHADOW)
        text_rect = text_surf.get_rect()
        if align == "topright":
            text_rect.topright = pos
        elif align == "topleft":
            text_rect.topleft = pos
        elif align == "center":
            text_rect.center = pos
        self.screen.blit(text_shadow, (text_rect.x + 2, text_rect.y + 2))
        self.screen.blit(text_surf, text_rect)

    def _draw_glowing_circle(self, surface, color, center, radius, transparent=False):
        glow_radius = int(radius * 1.8)
        glow_alpha = 70 if transparent else 100
        
        # Create a temporary surface for the glow
        temp_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(temp_surf, glow_radius, glow_radius, glow_radius, color + (glow_alpha,))
        
        center_int = (int(center[0]), int(center[1]))
        surface.blit(temp_surf, (center_int[0] - glow_radius, center_int[1] - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)
        pygame.gfxdraw.filled_circle(surface, center_int[0], center_int[1], int(radius), color)
        pygame.gfxdraw.aacircle(surface, center_int[0], center_int[1], int(radius), color)

    def _create_particles(self, pos, color, count, p_type):
        for _ in range(count):
            if p_type == 'strike':
                angle = self.np_random.uniform(0, math.pi * 2)
                speed = self.np_random.uniform(2, 6)
                vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            else: # out
                vel = [self.np_random.uniform(0.5, 1.5), self.np_random.uniform(-3, -1)]

            self.particles.append({
                'pos': list(pos),
                'vel': vel,
                'life': self.np_random.integers(20, 40),
                'max_life': 40,
                'color': color,
                'size': self.np_random.uniform(2, 5)
            })

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]

    def close(self):
        pygame.quit()


# Example usage:
if __name__ == '__main__':
    # This part is for interactive testing and will not be part of the final environment
    # It requires a display, so we unset the dummy video driver.
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv(render_mode="rgb_array")
    
    obs, info = env.reset()
    terminated = False
    
    pygame.display.set_caption("Idle Baseball Color Match")
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    clock = pygame.time.Clock()
    
    action = env.action_space.sample()
    action[0] = 0 # No movement
    action[1] = 0 # Space released
    action[2] = 0 # Shift released
    
    running = True
    while running:
        space_pressed_this_frame = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    space_pressed_this_frame = True
                if event.key == pygame.K_r: # Reset env
                    obs, info = env.reset()
        
        # For manual play, we simulate the 'held' action for one frame on key press
        action[1] = 1 if space_pressed_this_frame else 0

        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            print(f"Episode finished. Final Info: {info}")
            obs, info = env.reset()

        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Run at 30 FPS

    env.close()