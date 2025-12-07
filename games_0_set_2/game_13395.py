import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T21:16:38.973961
# Source Brief: brief_03395.md
# Brief Index: 3395
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A rhythm-based arcade game where the player controls a cursor to collect
    glowing orbs that appear in sync with a beat. The game's tempo and
    difficulty increase with each level. The goal is to collect orbs to score
    points and complete all five levels.

    The environment prioritizes visual quality and "game feel" with smooth
    animations, particle effects, and a minimalist neon aesthetic.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "A rhythm-based arcade game where the player controls a cursor to collect "
        "glowing orbs that appear in sync with a beat."
    )
    user_guide = (
        "Use the arrow keys (↑↓←→) to move the cursor and collect the orbs as they appear on the beat."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30  # Assumed FPS for consistent physics
        self.MAX_LEVELS = 5
        self.ORBS_PER_LEVEL = 100
        self.MAX_STEPS = 5000

        # --- Visuals & Colors ---
        self.COLOR_BG = (10, 15, 30)
        self.COLOR_GRID = (20, 30, 50)
        self.COLOR_CURSOR = (255, 0, 150)
        self.COLOR_CURSOR_GLOW = (150, 0, 80)
        self.COLOR_ORB = (0, 255, 255)
        self.COLOR_ORB_GLOW = (0, 150, 150)
        self.COLOR_BEAT_INDICATOR = (60, 70, 90)
        self.COLOR_BEAT_PULSE = (120, 130, 150)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_PARTICLE = self.COLOR_ORB

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_ui = pygame.font.SysFont("Consolas", 24)
            self.font_level = pygame.font.SysFont("Consolas", 32, bold=True)
        except pygame.error:
            self.font_ui = pygame.font.Font(None, 28)
            self.font_level = pygame.font.Font(None, 36)

        # --- Game State Variables (initialized in reset) ---
        self.steps = None
        self.score = None
        self.level = None
        self.game_over = None
        self.cursor_pos = None
        self.bpm = None
        self.beat_duration_frames = None
        self.beat_timer = None
        self.orbs_per_beat = None
        self.orbs = None
        self.particles = None

        # --- Initial Reset ---
        # self.reset() # reset is called by the wrapper/runner

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.level = 1
        self.game_over = False

        self.cursor_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=np.float32)
        self.orbs = []
        self.particles = []

        self._setup_level()

        return self._get_observation(), self._get_info()

    def _setup_level(self):
        """Initializes settings for the current level."""
        self.score = 0
        self.bpm = 60 * (1.2 ** (self.level - 1))
        self.beat_duration_frames = int((60 / self.bpm) * self.FPS)
        self.beat_timer = 0
        self.orbs_per_beat = self.level

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0

        # --- Handle Actions ---
        self._handle_action(action)

        # --- Update Game State ---
        self._update_beat()
        self._update_particles()

        # --- Process Beat Event ---
        if self.beat_timer == 0:
            # SFX: Beat pulse
            reward += self._spawn_and_collect_orbs()

        # --- Process Level Progression ---
        if self.score >= self.ORBS_PER_LEVEL:
            # SFX: Level up
            self.level += 1
            if self.level > self.MAX_LEVELS:
                self.game_over = True
                reward += 100  # Final completion reward
            else:
                reward += 10  # Level completion reward
                self._setup_level()

        # --- Check Termination ---
        terminated = self.game_over
        truncated = self.steps >= self.MAX_STEPS

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info(),
        )

    def _handle_action(self, action):
        """Processes the agent's action."""
        movement = action[0]
        # space_held = action[1] == 1  # Not used
        # shift_held = action[2] == 1  # Not used

        CURSOR_SPEED = 10.0
        if movement == 1:  # Up
            self.cursor_pos[1] -= CURSOR_SPEED
        elif movement == 2:  # Down
            self.cursor_pos[1] += CURSOR_SPEED
        elif movement == 3:  # Left
            self.cursor_pos[0] -= CURSOR_SPEED
        elif movement == 4:  # Right
            self.cursor_pos[0] += CURSOR_SPEED

        # Clamp cursor position to screen bounds
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.WIDTH)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.HEIGHT)

    def _update_beat(self):
        """Increments the beat timer."""
        self.beat_timer = (self.beat_timer + 1) % self.beat_duration_frames
        # Orbs are only visible for a fraction of the beat
        orb_lifetime_frames = max(3, self.beat_duration_frames // 4)
        for orb in self.orbs:
            orb['lifetime'] -= 1
        self.orbs = [orb for orb in self.orbs if orb['lifetime'] > 0]


    def _spawn_and_collect_orbs(self):
        """Spawns new orbs and checks for collection simultaneously."""
        self.orbs.clear()
        collected_count = 0
        CURSOR_RADIUS = 12
        ORB_RADIUS = 10

        for _ in range(self.orbs_per_beat):
            orb_pos = np.array([
                self.np_random.integers(ORB_RADIUS, self.WIDTH - ORB_RADIUS),
                self.np_random.integers(ORB_RADIUS, self.HEIGHT - ORB_RADIUS)
            ])
            distance = np.linalg.norm(self.cursor_pos - orb_pos)

            if distance < CURSOR_RADIUS + ORB_RADIUS:
                # SFX: Orb collect
                self.score += 1
                collected_count += 1
                self._spawn_particles(orb_pos)
            else:
                orb_lifetime_frames = max(3, self.beat_duration_frames // 4)
                self.orbs.append({'pos': orb_pos, 'lifetime': orb_lifetime_frames})

        return collected_count # This is the reward for this step

    def _update_particles(self):
        """Updates position and lifetime of all particles."""
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifetime'] -= 1
            p['vel'] *= 0.95 # Damping
        self.particles = [p for p in self.particles if p['lifetime'] > 0]

    def _spawn_particles(self, pos):
        """Creates an explosion of particles at a given position."""
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'lifetime': self.np_random.integers(10, 20),
                'size': self.np_random.uniform(1, 4)
            })

    def _get_observation(self):
        """Renders the game state to the screen and returns it as a numpy array."""
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_beat_indicator()
        self._render_orbs()
        self._render_particles()
        self._render_cursor()
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "level": self.level,
            "steps": self.steps,
        }

    # --- Rendering Helper Methods ---

    def _render_background(self):
        """Draws a static grid on the background."""
        for x in range(0, self.WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT), 1)
        for y in range(0, self.HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y), 1)

    def _render_beat_indicator(self):
        """Draws a central circle that pulses with the beat."""
        center = (self.WIDTH // 2, self.HEIGHT // 2)
        # Smooth pulse using a sine wave, peaking halfway through the beat
        pulse_progress = self.beat_timer / self.beat_duration_frames
        pulse_factor = (math.sin(pulse_progress * math.pi))
        
        # Base indicator
        base_radius = 50
        pygame.gfxdraw.aacircle(self.screen, center[0], center[1], base_radius, self.COLOR_BEAT_INDICATOR)
        
        # Pulsing wave
        pulse_radius = int(base_radius + pulse_factor * 30)
        pulse_alpha = int(150 * (1 - pulse_factor))
        if pulse_alpha > 0:
            pulse_color = (*self.COLOR_BEAT_PULSE, pulse_alpha)
            # This requires a new surface for alpha blending
            s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            pygame.gfxdraw.aacircle(s, center[0], center[1], pulse_radius, pulse_color)
            pygame.gfxdraw.filled_circle(s, center[0], center[1], pulse_radius, pulse_color)
            self.screen.blit(s, (0,0))


    def _render_orbs(self):
        """Draws all active orbs."""
        ORB_RADIUS = 10
        GLOW_RADIUS = 15
        for orb in self.orbs:
            pos_int = (int(orb['pos'][0]), int(orb['pos'][1]))
            # Draw glow
            self._draw_glow_circle(pos_int, GLOW_RADIUS, (*self.COLOR_ORB_GLOW, 100))
            # Draw orb
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], ORB_RADIUS, self.COLOR_ORB)
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], ORB_RADIUS, self.COLOR_ORB)

    def _render_particles(self):
        """Draws all active particles."""
        for p in self.particles:
            pos_int = (int(p['pos'][0]), int(p['pos'][1]))
            size = int(p['size'] * (p['lifetime'] / 20.0))
            if size > 0:
                alpha = int(255 * (p['lifetime'] / 20.0))
                color = (*self.COLOR_PARTICLE, alpha)
                s = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.circle(s, color, (size, size), size)
                self.screen.blit(s, (pos_int[0]-size, pos_int[1]-size), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_cursor(self):
        """Draws the player's cursor with a glow effect."""
        CURSOR_RADIUS = 12
        GLOW_RADIUS = 22
        pos_int = (int(self.cursor_pos[0]), int(self.cursor_pos[1]))
        
        # Draw glow
        self._draw_glow_circle(pos_int, GLOW_RADIUS, (*self.COLOR_CURSOR_GLOW, 100))
        # Draw cursor
        pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], CURSOR_RADIUS, self.COLOR_CURSOR)
        pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], CURSOR_RADIUS, self.COLOR_CURSOR)

    def _render_ui(self):
        """Draws the score and level information."""
        score_text = self.font_ui.render(f"SCORE: {self.score}/{self.ORBS_PER_LEVEL}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        level_text_str = f"LEVEL: {self.level}/{self.MAX_LEVELS}"
        if self.game_over and self.level > self.MAX_LEVELS:
            level_text_str = "COMPLETE!"

        level_text = self.font_ui.render(level_text_str, True, self.COLOR_TEXT)
        self.screen.blit(level_text, (self.WIDTH - level_text.get_width() - 10, 10))

    def _draw_glow_circle(self, pos, radius, color):
        """Helper to draw a semi-transparent circle for a glow effect."""
        s = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(s, color, (radius, radius), radius)
        self.screen.blit(s, (pos[0] - radius, pos[1] - radius), special_flags=pygame.BLEND_RGBA_ADD)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
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
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)

        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # --- Example Usage and Manual Play ---
    # Set the video driver for local rendering
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Pygame setup for manual play
    render_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Rhythm Orb Collector")
    clock = pygame.time.Clock()
    
    total_reward = 0
    
    print("\n--- Manual Control ---")
    print(GameEnv.user_guide)
    print("Q: Quit")
    
    while not done:
        # --- Action Mapping for Manual Play ---
        movement = 0 # No-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        # Space and Shift are not used in this game
        space_held = 0
        shift_held = 0
        
        action = [movement, space_held, shift_held]

        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

        # --- Rendering for Display ---
        # The observation is already the rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling (for quitting) ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                done = True
        
        clock.tick(env.FPS)

    print(f"\nGame Over!")
    print(f"Final Info: {info}")
    print(f"Total Reward: {total_reward}")
    
    env.close()