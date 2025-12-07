import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:28:09.999823
# Source Brief: brief_01081.md
# Brief Index: 1081
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A musical dexterity game where the agent controls two "hands" to pluck notes on a staff.
    The goal is to score points by plucking notes, with a bonus for plucking with both
    hands simultaneously. Each "round" of notes has a 3-second timer.

    Action Space: MultiDiscrete([5, 2, 2])
    - actions[0]: Movement
        - 0: No-op
        - 1: Left Hand Left
        - 2: Left Hand Right
        - 3: Right Hand Left
        - 4: Right Hand Right
    - actions[1]: Pluck with Left Hand (0=released, 1=pressed)
    - actions[2]: Pluck with Right Hand (0=released, 1=pressed)

    Observation Space: Box(shape=(400, 640, 3), dtype=uint8)
    - An RGB image of the game screen.

    Reward Structure:
    - +1.0 for each correctly plucked note.
    - +5.0 bonus for plucking a note with each hand on the same frame (synchronization).
    - -0.1 for each note remaining when the round timer expires.
    - +100.0 for winning the game (reaching the score goal).
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "A musical dexterity game where you control two hands to pluck notes as they appear. "
        "Score points by hitting notes in time and earn bonuses for synchronized plucks."
    )
    user_guide = (
        "Controls: Use A/D to move the left hand and ←→ arrows to move the right hand. "
        "Press space to pluck with the left hand and shift to pluck with the right hand."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    MAX_EPISODE_STEPS = 3000
    SCORE_GOAL = 1000
    ROUND_DURATION_SECONDS = 3.0

    # Colors
    COLOR_BG_START = (15, 20, 35)
    COLOR_BG_END = (30, 40, 70)
    COLOR_STAFF = (80, 100, 140)
    COLOR_NOTE = (255, 220, 0) # Bright Yellow
    COLOR_NOTE_GLOW = (255, 220, 0, 50)
    COLOR_HAND_LEFT = (0, 150, 255) # Bright Blue
    COLOR_HAND_RIGHT = (255, 50, 100) # Bright Pink/Red
    COLOR_HAND_GLOW = (150, 200, 255, 30)
    COLOR_SUCCESS = (50, 255, 50) # Green
    COLOR_FAIL = (255, 50, 50) # Red
    COLOR_TEXT = (240, 240, 255)
    COLOR_TIMER_BAR = (0, 200, 220)

    # Game Mechanics
    NUM_STAFF_LINES = 5
    NUM_NOTE_POSITIONS_PER_SIDE = 10
    NUM_NOTES_PER_ROUND = 10
    HAND_SPEED = 0.35  # Interpolation factor for smooth movement
    HAND_SIZE = 18
    NOTE_RADIUS = 10

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Gymnasium spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_popup = pygame.font.SysFont("Verdana", 20, bold=True)
        
        # --- Game State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False

        # Hand positions
        self.hand_pos_idx_left = 0
        self.hand_pos_idx_right = 0
        self.visual_hand_x_left = 0.0
        self.visual_hand_x_right = 0.0

        # Notes and timing
        self.notes = []
        self.round_timer = 0
        self.round_max_time = self.ROUND_DURATION_SECONDS * self.FPS

        # Visual effects
        self.particles = []
        self.popup_texts = []

        # Staff layout calculation
        self._calculate_layout()
        
        # Initial state is set in reset
        # self.reset() # This will be called by the wrapper/runner

    def _calculate_layout(self):
        """Pre-calculate positions for staff lines and note slots."""
        self.staff_y_positions = np.linspace(
            self.SCREEN_HEIGHT * 0.3, self.SCREEN_HEIGHT * 0.7, self.NUM_STAFF_LINES
        ).astype(int)

        self.pluck_zone_left = (self.SCREEN_WIDTH * 0.1, self.SCREEN_WIDTH * 0.45)
        self.pluck_zone_right = (self.SCREEN_WIDTH * 0.55, self.SCREEN_WIDTH * 0.9)
        
        self.note_x_positions_left = np.linspace(
            self.pluck_zone_left[0], self.pluck_zone_left[1], self.NUM_NOTE_POSITIONS_PER_SIDE
        ).astype(int)
        
        self.note_x_positions_right = np.linspace(
            self.pluck_zone_right[0], self.pluck_zone_right[1], self.NUM_NOTE_POSITIONS_PER_SIDE
        ).astype(int)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # Reset visual effects
        self.particles = []
        self.popup_texts = []
        
        # Start the first round
        self._start_new_round()
        
        # Center hands visually and logically
        center_idx = self.NUM_NOTE_POSITIONS_PER_SIDE // 2
        self.hand_pos_idx_left = center_idx
        self.hand_pos_idx_right = center_idx
        self.visual_hand_x_left = self.note_x_positions_left[center_idx]
        self.visual_hand_x_right = self.note_x_positions_right[center_idx]
        
        return self._get_observation(), self._get_info()

    def _start_new_round(self):
        """Initializes a new set of notes and resets the round timer."""
        self.round_timer = self.round_max_time
        self.notes = []
        
        # Generate possible note slots: (side, x_idx, y_idx) where side is 0 for left, 1 for right
        possible_slots = []
        for side in range(2):
            for x_idx in range(self.NUM_NOTE_POSITIONS_PER_SIDE):
                for y_idx in range(self.NUM_STAFF_LINES):
                    possible_slots.append((side, x_idx, y_idx))
        
        # Randomly choose slots for the new notes
        chosen_indices = self.np_random.choice(len(possible_slots), self.NUM_NOTES_PER_ROUND, replace=False)
        
        for slot_idx in chosen_indices:
            side, x_idx, y_idx = possible_slots[slot_idx]
            if side == 0: # Left side
                x = self.note_x_positions_left[x_idx]
            else: # Right side
                x = self.note_x_positions_right[x_idx]
            y = self.staff_y_positions[y_idx]
            
            self.notes.append({
                "pos": (x, y),
                "side": side,
                "x_idx": x_idx,
                "plucked": False,
                "initial_life": 1.0 # For visual effects
            })

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0.0

        # --- 1. Unpack and handle actions ---
        movement, pluck_left_action, pluck_right_action = action
        
        # --- Handle Movement ---
        if movement == 1: self.hand_pos_idx_left -= 1
        elif movement == 2: self.hand_pos_idx_left += 1
        elif movement == 3: self.hand_pos_idx_right -= 1
        elif movement == 4: self.hand_pos_idx_right += 1
        
        # Clamp hand positions
        self.hand_pos_idx_left = max(0, min(self.NUM_NOTE_POSITIONS_PER_SIDE - 1, self.hand_pos_idx_left))
        self.hand_pos_idx_right = max(0, min(self.NUM_NOTE_POSITIONS_PER_SIDE - 1, self.hand_pos_idx_right))

        # --- Handle Plucking ---
        left_pluck_success = False
        right_pluck_success = False

        if pluck_left_action == 1:
            for note in self.notes:
                if not note["plucked"] and note["side"] == 0 and note["x_idx"] == self.hand_pos_idx_left:
                    note["plucked"] = True
                    left_pluck_success = True
                    reward += 1.0
                    self.score += 10
                    self._create_particles(note["pos"], self.COLOR_SUCCESS, 20)
                    self._create_popup_text(f"+10", note["pos"], self.COLOR_SUCCESS)
                    break
            if not left_pluck_success:
                pos = (self.note_x_positions_left[self.hand_pos_idx_left], self.SCREEN_HEIGHT // 2)
                self._create_particles(pos, self.COLOR_FAIL, 5, 0.5)

        if pluck_right_action == 1:
            for note in self.notes:
                if not note["plucked"] and note["side"] == 1 and note["x_idx"] == self.hand_pos_idx_right:
                    note["plucked"] = True
                    right_pluck_success = True
                    reward += 1.0
                    self.score += 10
                    self._create_particles(note["pos"], self.COLOR_SUCCESS, 20)
                    self._create_popup_text(f"+10", note["pos"], self.COLOR_SUCCESS)
                    break
            if not right_pluck_success:
                pos = (self.note_x_positions_right[self.hand_pos_idx_right], self.SCREEN_HEIGHT // 2)
                self._create_particles(pos, self.COLOR_FAIL, 5, 0.5)
        
        # Synchronization Bonus
        if left_pluck_success and right_pluck_success:
            reward += 5.0
            self.score += 50
            sync_pos = (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT * 0.8)
            self._create_popup_text("SYNC! +50", sync_pos, self.COLOR_NOTE, 2 * self.FPS)

        # --- 2. Update Game State ---
        self.round_timer -= 1

        # Interpolate visual hand positions for smoothness
        target_x_left = self.note_x_positions_left[self.hand_pos_idx_left]
        target_x_right = self.note_x_positions_right[self.hand_pos_idx_right]
        self.visual_hand_x_left += (target_x_left - self.visual_hand_x_left) * self.HAND_SPEED
        self.visual_hand_x_right += (target_x_right - self.visual_hand_x_right) * self.HAND_SPEED

        # Update visual effects
        self._update_particles()
        self._update_popup_texts()
        for note in self.notes:
            if note["plucked"]:
                note["initial_life"] = max(0, note["initial_life"] - 0.1)

        # Check for round end
        all_notes_plucked = all(n["plucked"] for n in self.notes)
        if self.round_timer <= 0 or all_notes_plucked:
            if not all_notes_plucked:
                unplucked_count = sum(1 for n in self.notes if not n["plucked"])
                reward -= 0.1 * unplucked_count
            self._start_new_round()

        # --- 3. Check for Termination ---
        terminated = (self.score >= self.SCORE_GOAL)
        truncated = (self.steps >= self.MAX_EPISODE_STEPS)
        if terminated or truncated:
            self.game_over = True
            if self.score >= self.SCORE_GOAL:
                reward += 100.0
        
        # --- 4. Return step information ---
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _get_observation(self):
        self._render_all()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "round_time_left": self.round_timer / self.FPS,
        }

    # --- Rendering Methods ---

    def _render_all(self):
        """Master render function."""
        self._draw_background()
        self._draw_staff()
        self._draw_notes()
        self._draw_hands()
        self._draw_particles()
        self._draw_popup_texts()
        self._draw_ui()

    def _draw_background(self):
        """Draw a vertical gradient for the background."""
        for y in range(self.SCREEN_HEIGHT):
            ratio = y / self.SCREEN_HEIGHT
            color = (
                int(self.COLOR_BG_START[0] * (1 - ratio) + self.COLOR_BG_END[0] * ratio),
                int(self.COLOR_BG_START[1] * (1 - ratio) + self.COLOR_BG_END[1] * ratio),
                int(self.COLOR_BG_START[2] * (1 - ratio) + self.COLOR_BG_END[2] * ratio),
            )
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))

    def _draw_staff(self):
        """Draw the horizontal staff lines."""
        for y in self.staff_y_positions:
            pygame.draw.line(self.screen, self.COLOR_STAFF, (0, y), (self.SCREEN_WIDTH, y), 2)

    def _draw_notes(self):
        """Draw the notes on the staff."""
        for note in self.notes:
            if note["initial_life"] > 0:
                self._draw_glowing_circle(
                    self.screen,
                    note["pos"],
                    self.NOTE_RADIUS,
                    self.COLOR_NOTE,
                    glow_color=(self.COLOR_NOTE_GLOW[0], self.COLOR_NOTE_GLOW[1], self.COLOR_NOTE_GLOW[2], int(self.COLOR_NOTE_GLOW[3] * note["initial_life"]))
                )
    
    def _draw_hands(self):
        """Draw the player's hands as glowing triangles."""
        # Left Hand
        y_center = self.SCREEN_HEIGHT // 2
        p_left = [
            (int(self.visual_hand_x_left), y_center - self.HAND_SIZE // 2),
            (int(self.visual_hand_x_left - self.HAND_SIZE // 2), y_center + self.HAND_SIZE // 2),
            (int(self.visual_hand_x_left + self.HAND_SIZE // 2), y_center + self.HAND_SIZE // 2),
        ]
        self._draw_glowing_polygon(self.screen, p_left, self.COLOR_HAND_LEFT, self.COLOR_HAND_GLOW)
        
        # Right Hand
        p_right = [
            (int(self.visual_hand_x_right), y_center - self.HAND_SIZE // 2),
            (int(self.visual_hand_x_right - self.HAND_SIZE // 2), y_center + self.HAND_SIZE // 2),
            (int(self.visual_hand_x_right + self.HAND_SIZE // 2), y_center + self.HAND_SIZE // 2),
        ]
        self._draw_glowing_polygon(self.screen, p_right, self.COLOR_HAND_RIGHT, self.COLOR_HAND_GLOW)


    def _draw_ui(self):
        """Draw score and timer."""
        # Score
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Timer bar
        timer_ratio = max(0, self.round_timer / self.round_max_time)
        bar_width = 150
        bar_height = 20
        bar_x = self.SCREEN_WIDTH - bar_width - 10
        bar_y = 10
        pygame.draw.rect(self.screen, self.COLOR_STAFF, (bar_x, bar_y, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_TIMER_BAR, (bar_x, bar_y, bar_width * timer_ratio, bar_height))

    # --- Visual Effects ---

    def _create_particles(self, pos, color, count, speed_multiplier=1.0):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4) * speed_multiplier
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = self.np_random.integers(self.FPS // 2, self.FPS)
            size = self.np_random.uniform(2, 5)
            self.particles.append({"pos": list(pos), "vel": vel, "life": life, "max_life": life, "color": color, "size": size})

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)

    def _draw_particles(self):
        for p in self.particles:
            life_ratio = p["life"] / p["max_life"]
            color = (p["color"][0], p["color"][1], p["color"][2], int(255 * life_ratio))
            size = int(p["size"] * life_ratio)
            if size > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p["pos"][0]), int(p["pos"][1]), size, color)

    def _create_popup_text(self, text, pos, color, life=None):
        if life is None:
            life = self.FPS
        self.popup_texts.append({"text": text, "pos": list(pos), "life": life, "max_life": life, "color": color})

    def _update_popup_texts(self):
        for pt in self.popup_texts[:]:
            pt["pos"][1] -= 1 # Move up
            pt["life"] -= 1
            if pt["life"] <= 0:
                self.popup_texts.remove(pt)
    
    def _draw_popup_texts(self):
        for pt in self.popup_texts:
            life_ratio = pt["life"] / pt["max_life"]
            alpha = int(255 * (life_ratio ** 0.5))
            text_surf = self.font_popup.render(pt["text"], True, pt["color"])
            text_surf.set_alpha(alpha)
            text_rect = text_surf.get_rect(center=pt["pos"])
            self.screen.blit(text_surf, text_rect)

    @staticmethod
    def _draw_glowing_circle(surface, pos, radius, color, glow_color):
        pos_int = (int(pos[0]), int(pos[1]))
        for i in range(4):
            r = radius + i * 2
            alpha = glow_color[3] // (i + 1)
            pygame.gfxdraw.aacircle(surface, pos_int[0], pos_int[1], r, (glow_color[0], glow_color[1], glow_color[2], alpha))
        pygame.gfxdraw.filled_circle(surface, pos_int[0], pos_int[1], radius, color)
        pygame.gfxdraw.aacircle(surface, pos_int[0], pos_int[1], radius, color)

    @staticmethod
    def _draw_glowing_polygon(surface, points, color, glow_color):
        # This is a simplified glow effect. For a real blur, you'd need multiple surfaces.
        # Here we just draw widening, semi-transparent polygons.
        try:
            pygame.gfxdraw.aapolygon(surface, points, color)
            pygame.gfxdraw.filled_polygon(surface, points, color)
        except: # gfxdraw can fail on certain degenerate polygons
            pygame.draw.polygon(surface, color, points)


    def close(self):
        pygame.quit()


# --- Example Usage ---
if __name__ == '__main__':
    # This block is for human play and debugging.
    # It will not be run by the evaluation system.
    # Ensure that the environment works correctly in headless mode.
    
    # To run with display, comment out the os.environ line at the top
    # and uncomment the display-related lines in this block.
    
    # Forcing display for local testing:
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv()
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Musical Dexterity")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        # --- Human Controls ---
        movement = 0 # no-op
        pluck_left = 0
        pluck_right = 0

        # Check for key presses for actions that happen once
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0
                if event.key == pygame.K_SPACE:
                    pluck_left = 1
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                    pluck_right = 1

        # Check for held keys for continuous movement
        keys = pygame.key.get_pressed()
        if keys[pygame.K_a]: movement = 1 # Left hand left
        elif keys[pygame.K_d]: movement = 2 # Left hand right
        elif keys[pygame.K_LEFT]: movement = 3 # Right hand left
        elif keys[pygame.K_RIGHT]: movement = 4 # Right hand right

        action = [movement, pluck_left, pluck_right]
        
        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            obs, info = env.reset()
            total_reward = 0

        # --- Rendering ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.FPS)
        
    env.close()