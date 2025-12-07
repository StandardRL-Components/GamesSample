import os
import math
import random
import os
import pygame


# Set the SDL video driver to "dummy" BEFORE importing pygame.
# This is crucial for running Pygame in a headless environment (e.g., on a server).
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import gymnasium as gym
import numpy as np
import pygame
import pygame.gfxdraw
from gymnasium.spaces import MultiDiscrete


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Press space to slice the notes as they cross the beat line."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Slice falling notes on the beat in an isometric-2D rhythm game to achieve a high score."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    # Colors
    COLOR_BG = (15, 18, 32)
    COLOR_GRID = (30, 35, 60)
    COLOR_BEAT_LINE = (255, 255, 255)
    COLOR_BEAT_LINE_GLOW = (180, 180, 255)
    COLOR_NOTE_LANES = [
        (0, 255, 255),  # Cyan
        (255, 0, 255),  # Magenta
        (255, 255, 0),  # Yellow
        (0, 255, 0),    # Green
    ]
    COLOR_TEXT = (220, 220, 240)
    COLOR_HIT_PERFECT = (255, 255, 100)
    COLOR_HIT_GOOD = (150, 255, 255)
    COLOR_MISS_X = (255, 50, 50)
    COLOR_MISS_ICON_EMPTY = (60, 60, 80)
    COLOR_MISS_ICON_FULL = (200, 40, 40)

    # Screen and World Dimensions
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    ISO_ORIGIN_X = SCREEN_WIDTH // 2
    ISO_ORIGIN_Y = 80
    LANE_WIDTH = 40
    NUM_LANES = 4
    WORLD_DEPTH = 200

    # Gameplay Parameters
    BEAT_LINE_DEPTH = 150
    HIT_WINDOW_PERFECT = 4
    HIT_WINDOW_GOOD = 10
    MISS_LINE_DEPTH = BEAT_LINE_DEPTH + 15
    INITIAL_NOTE_SPEED = 1.5
    INITIAL_SPAWN_PROB = 0.03
    MAX_STEPS = 1000
    WIN_SCORE = 500
    MAX_MISSES = 5

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        # A display mode must be set for pygame to work, even in headless mode.
        # This is required for functions like font rendering and surface conversions.
        pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 16)
        
        self.notes = []
        self.particles = []
        self.feedback_text = []

        # self.reset() is called here to ensure the state is initialized.
        # It's called again by the typical gym loop, which is fine.
        self.reset()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.misses = 0
        self.game_over = False
        
        self.notes = []
        self.particles = []
        self.feedback_text = []

        self.note_speed = self.INITIAL_NOTE_SPEED
        self.note_spawn_prob = self.INITIAL_SPAWN_PROB
        
        self.last_space_state = False
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        movement, space_binary, shift_binary = action
        space_pressed = space_binary == 1 and not self.last_space_state
        self.last_space_state = space_binary == 1

        reward = 0.0
        
        self._update_game_state()

        if space_pressed:
            # sfx: slice_swing.wav
            reward += self._handle_slice()

        reward += self._check_missed_notes()
        
        self.steps += 1
        
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS

        if terminated and not truncated: # Game-specific termination (win/loss)
            if self.score >= self.WIN_SCORE:
                reward += 100
                # sfx: win_jingle.wav
            elif self.misses >= self.MAX_MISSES:
                reward -= 50
                # sfx: lose_sound.wav
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _update_game_state(self):
        # Update and remove old particles
        self.particles = [p for p in self.particles if p.update()]
        self.feedback_text = [f for f in self.feedback_text if f.update()]

        # Update note positions
        for note in self.notes:
            note['depth'] += self.note_speed

        # Spawn new notes
        if self.np_random.random() < self.note_spawn_prob:
            self._spawn_note()

        # Scale difficulty
        if self.steps > 0 and self.steps % 100 == 0:
            self.note_speed = min(self.note_speed + 0.05, 5.0)
        if self.steps > 0 and self.steps % 200 == 0:
            self.note_spawn_prob = min(self.note_spawn_prob + 0.005, 0.2)
            
    def _handle_slice(self):
        reward = 0.0
        hit_something = False
        
        # We check notes in reverse to allow safe removal while iterating
        for note in reversed(self.notes):
            distance = abs(note['depth'] - self.BEAT_LINE_DEPTH)
            
            if distance <= self.HIT_WINDOW_GOOD:
                hit_something = True
                
                if distance <= self.HIT_WINDOW_PERFECT:
                    # sfx: hit_perfect.wav
                    self.score += 10
                    reward += 5.0
                    self._create_particles(note, self.COLOR_HIT_PERFECT, 30)
                    self._create_feedback_text("PERFECT!", note, self.COLOR_HIT_PERFECT)
                else:
                    # sfx: hit_good.wav
                    self.score += 5
                    reward += 1.0
                    self._create_particles(note, self.COLOR_HIT_GOOD, 15)
                    self._create_feedback_text("GOOD", note, self.COLOR_HIT_GOOD)
                
                self.notes.remove(note)
        
        if not hit_something:
            # sfx: miss_swing.wav
            reward -= 1.0 # Penalty for slicing at the wrong time
            
        return reward

    def _check_missed_notes(self):
        reward = 0.0
        notes_to_remove = []
        for note in self.notes:
            if note['depth'] > self.MISS_LINE_DEPTH:
                # sfx: note_miss.wav
                self.misses += 1
                reward -= 1.0
                notes_to_remove.append(note)
                self._create_feedback_text("MISS", note, self.COLOR_MISS_X)

        if notes_to_remove:
            self.notes = [n for n in self.notes if n not in notes_to_remove]
        
        return reward

    def _spawn_note(self):
        lane = self.np_random.integers(0, self.NUM_LANES)
        # Prevent notes from spawning in the same lane too close together
        if any(n['lane'] == lane and n['depth'] < 30 for n in self.notes):
            return
            
        self.notes.append({
            'lane': lane,
            'depth': 0,
            'color': self.COLOR_NOTE_LANES[lane]
        })

    def _check_termination(self):
        return (
            self.score >= self.WIN_SCORE or 
            self.misses >= self.MAX_MISSES
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        # Pygame array is (width, height, channels). Obs space is (height, width, channels).
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "misses": self.misses
        }

    # --- Rendering ---

    def _iso_project(self, lane, depth, height=0):
        # Center the lanes
        lane_centered = lane - (self.NUM_LANES - 1) / 2.0
        # Project to screen coordinates
        screen_x = self.ISO_ORIGIN_X + lane_centered * self.LANE_WIDTH * 0.707
        screen_y = self.ISO_ORIGIN_Y + depth * 0.5 - height
        return int(screen_x), int(screen_y)

    def _render_game(self):
        self._render_grid()
        self._render_beat_line()
        
        # Sort notes by depth for correct Z-ordering
        sorted_notes = sorted(self.notes, key=lambda n: n['depth'])
        
        for note in sorted_notes:
            self._render_note(note)
        
        for p in self.particles:
            p.draw(self.screen)
            
        for f in self.feedback_text:
            f.draw(self.screen)

    def _render_grid(self):
        # Horizontal lines
        for i in range(0, self.WORLD_DEPTH + 1, 20):
            p1 = self._iso_project(-0.5, i)
            p2 = self._iso_project(self.NUM_LANES - 0.5, i)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, p1, p2)
        # Vertical lines
        for i in range(self.NUM_LANES):
            p1 = self._iso_project(i - 0.5, 0)
            p2 = self._iso_project(i - 0.5, self.WORLD_DEPTH)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, p1, p2)

    def _render_beat_line(self):
        # Pulsing glow effect
        glow_size = int(5 + 3 * math.sin(self.steps * 0.2))
        p1_glow = self._iso_project(-0.5, self.BEAT_LINE_DEPTH)
        p2_glow = self._iso_project(self.NUM_LANES - 0.5, self.BEAT_LINE_DEPTH)
        
        # Draw multiple lines for a soft glow - this is slow, use a different method if perf is an issue
        for i in range(glow_size, 0, -1):
            alpha = 80 * (1 - i / glow_size)
            glow_color = (*self.COLOR_BEAT_LINE_GLOW, int(alpha))
            # Create a temporary surface for alpha blending
            temp_surf = self.screen.convert_alpha()
            temp_surf.fill((0,0,0,0))
            pygame.draw.line(temp_surf, glow_color, (p1_glow[0] - i, p1_glow[1]), (p2_glow[0] + i, p2_glow[1]), 1)
            pygame.draw.line(temp_surf, glow_color, (p1_glow[0] + i, p1_glow[1]), (p2_glow[0] - i, p2_glow[1]), 1)
            pygame.draw.line(temp_surf, glow_color, (p1_glow[0], p1_glow[1] - i), (p2_glow[0], p2_glow[1] - i), 1)
            pygame.draw.line(temp_surf, glow_color, (p1_glow[0], p1_glow[1] + i), (p2_glow[0], p2_glow[1] + i), 1)
            self.screen.blit(temp_surf, (0,0))


        p1 = self._iso_project(-0.7, self.BEAT_LINE_DEPTH)
        p2 = self._iso_project(self.NUM_LANES - 0.7, self.BEAT_LINE_DEPTH)
        pygame.draw.line(self.screen, self.COLOR_BEAT_LINE, p1, p2, 3)

    def _render_note(self, note):
        note_width = self.LANE_WIDTH * 0.7
        note_height = self.LANE_WIDTH * 0.35
        note_thickness = 8

        # Calculate the 4 corners of the top face of the note
        p_center = self._iso_project(note['lane'], note['depth'])
        
        p1 = (p_center[0], p_center[1] - note_height / 2)
        p2 = (p_center[0] + note_width / 2, p_center[1])
        p3 = (p_center[0], p_center[1] + note_height / 2)
        p4 = (p_center[0] - note_width / 2, p_center[1])
        
        # Darker side color
        side_color = tuple(max(0, c - 80) for c in note['color'])
        
        # Draw side faces first for 3D effect
        pygame.draw.polygon(self.screen, side_color, [p3, p4, (p4[0], p4[1] + note_thickness), (p3[0], p3[1] + note_thickness)])
        pygame.draw.polygon(self.screen, side_color, [p2, p3, (p3[0], p3[1] + note_thickness), (p2[0], p2[1] + note_thickness)])
        
        # Draw top face
        top_face = [p1, p2, p3, p4]
        pygame.gfxdraw.aapolygon(self.screen, top_face, note['color'])
        pygame.gfxdraw.filled_polygon(self.screen, top_face, note['color'])

    def _render_ui(self):
        # Score display
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Misses display
        miss_text = self.font_small.render("MISSES:", True, self.COLOR_TEXT)
        self.screen.blit(miss_text, (self.SCREEN_WIDTH - 180, 15))
        for i in range(self.MAX_MISSES):
            color = self.COLOR_MISS_ICON_FULL if i < self.misses else self.COLOR_MISS_ICON_EMPTY
            pos = (self.SCREEN_WIDTH - 100 + i * 18, 22)
            pygame.draw.circle(self.screen, color, pos, 6)

    def _create_particles(self, note, color, count):
        pos = self._iso_project(note['lane'], note['depth'])
        for _ in range(count):
            self.particles.append(Particle(pos, color, self.np_random))
            
    def _create_feedback_text(self, text, note, color):
        pos = self._iso_project(note['lane'], self.BEAT_LINE_DEPTH - 20)
        self.feedback_text.append(FeedbackText(text, pos, color))

    def close(self):
        pygame.quit()

# --- Helper Classes for Effects ---

class Particle:
    def __init__(self, pos, color, rng):
        self.x, self.y = pos
        self.color = color
        angle = rng.random() * 2 * math.pi
        speed = rng.random() * 3 + 1
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.lifetime = rng.integers(15, 30)
        self.radius = rng.random() * 2 + 1

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.lifetime -= 1
        self.radius -= 0.05
        return self.lifetime > 0 and self.radius > 0

    def draw(self, surface):
        if self.radius > 0:
            pos = (int(self.x), int(self.y))
            pygame.draw.circle(surface, self.color, pos, int(self.radius))

class FeedbackText:
    def __init__(self, text, pos, color):
        self.font = pygame.font.SysFont("Verdana", 20, bold=True)
        self.text = text
        self.x, self.y = pos
        self.color = color
        self.lifetime = 30
        self.alpha = 255
        self.surface = self.font.render(text, True, color).convert_alpha()

    def update(self):
        self.y -= 1
        self.lifetime -= 1
        self.alpha = max(0, self.alpha - 8)
        return self.lifetime > 0

    def draw(self, surface):
        self.surface.set_alpha(self.alpha)
        text_rect = self.surface.get_rect(center=(int(self.x), int(self.y)))
        surface.blit(self.surface, text_rect)