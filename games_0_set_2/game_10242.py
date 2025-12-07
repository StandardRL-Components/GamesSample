import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:09:08.755448
# Source Brief: brief_00242.md
# Brief Index: 242
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class Particle:
    """A simple class for managing particles for visual effects."""
    def __init__(self, x, y, vx, vy, color, radius, lifetime):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.color = color
        self.radius = radius
        self.lifetime = lifetime
        self.initial_lifetime = lifetime

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.lifetime -= 1
        self.vx *= 0.98  # Damping
        self.vy *= 0.98

    def draw(self, surface):
        if self.lifetime > 0:
            alpha = int(255 * (self.lifetime / self.initial_lifetime))
            current_radius = int(self.radius * (self.lifetime / self.initial_lifetime))
            if current_radius > 0:
                # Use a temporary surface for alpha blending
                temp_surf = pygame.Surface((current_radius * 2, current_radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, (*self.color, alpha), (current_radius, current_radius), current_radius)
                surface.blit(temp_surf, (int(self.x - current_radius), int(self.y - current_radius)))


class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player solves logic circuit puzzles
    by placing dream fragments, while avoiding detection from a randomized
    subconscious system represented by dice rolls.
    """
    metadata = {"render_modes": ["rgb_array"]}
    game_description = (
        "Solve logic circuit puzzles by placing dream fragments on a grid. "
        "Avoid detection from the subconscious by managing risk and manipulating dice rolls."
    )
    user_guide = (
        "Controls: Use ↑↓←→ arrow keys to move the cursor. Press space to place a fragment and shift to manipulate the dice."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = Box(low=0, high=255, shape=(400, 640, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        self.WIDTH, self.HEIGHT = 640, 400
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_small = pygame.font.SysFont("Consolas", 16)
            self.font_medium = pygame.font.SysFont("Consolas", 24)
            self.font_large = pygame.font.SysFont("Consolas", 48)
        except pygame.error:
            self.font_small = pygame.font.SysFont("monospace", 16)
            self.font_medium = pygame.font.SysFont("monospace", 24)
            self.font_large = pygame.font.SysFont("monospace", 48)

        # --- Visuals & Colors ---
        self.COLOR_BG = (15, 10, 35)
        self.COLOR_GRID = (50, 40, 80)
        self.COLOR_GOAL = (255, 220, 0)
        self.COLOR_FRAGMENT = (0, 150, 255)
        self.COLOR_CURSOR = (0, 255, 150)
        self.COLOR_DETECTION = (255, 0, 50)
        self.COLOR_TEXT = (220, 220, 255)
        self.COLOR_DICE = (255, 255, 255)

        # --- Game Board ---
        self.BOARD_COLS, self.BOARD_ROWS = 12, 8
        self.CELL_SIZE = 36
        self.BOARD_OFFSET_X = 40
        self.BOARD_OFFSET_Y = 40
        self.GRID_WIDTH = self.BOARD_COLS * self.CELL_SIZE
        self.GRID_HEIGHT = self.BOARD_ROWS * self.CELL_SIZE

        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_state = False
        self.puzzles_solved_session = 0

        self.board = []
        self.puzzle_goal = []
        self.cursor_pos = [0, 0]
        self.visual_cursor_pos = [0.0, 0.0] # For smooth interpolation
        self.available_fragments = 0
        self.dice_manipulations_left = 0
        self.detection_prob = 0.0
        self.dice_roll = 1
        self.detection_modifier = 1.0

        self.last_space_held = False
        self.last_shift_held = False
        
        self.particles = []
        self.feedback_text = ""
        self.feedback_timer = 0
        self.dice_anim_timer = 0

        # self.reset() is called by the environment wrapper
        # self.validate_implementation() is not needed for production

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_state = False
        
        # In case of sequential calls to reset, we don't want to increment level indefinitely without playing
        # self.level = 1 + self.puzzles_solved_session // 3
        # A simpler logic is to reset puzzles solved count as well.
        # Or, we can handle it based on options if provided.
        if options and options.get("new_session", False):
            self.puzzles_solved_session = 0
        
        self.level = 1 + self.puzzles_solved_session // 3
        self._generate_puzzle()

        self.board = [[0 for _ in range(self.BOARD_COLS)] for _ in range(self.BOARD_ROWS)]
        self.cursor_pos = [self.BOARD_COLS // 2, self.BOARD_ROWS // 2]
        self.visual_cursor_pos = [float(self.cursor_pos[0]), float(self.cursor_pos[1])]
        self.available_fragments = len(self.puzzle_goal) + 2 # Extra fragments
        self.dice_manipulations_left = 2
        self.detection_prob = 0.10
        self.dice_roll = self.np_random.integers(1, 7)
        self.detection_modifier = 1.0

        self.last_space_held = False
        self.last_shift_held = False

        self.particles.clear()
        self.feedback_text = ""
        self.feedback_timer = 0
        self.dice_anim_timer = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        self.steps += 1
        
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.last_space_held
        shift_pressed = shift_held and not self.last_shift_held
        self.last_space_held, self.last_shift_held = space_held, shift_held

        # --- Handle Actions ---
        if movement == 1: self.cursor_pos[1] -= 1 # Up
        elif movement == 2: self.cursor_pos[1] += 1 # Down
        elif movement == 3: self.cursor_pos[0] -= 1 # Left
        elif movement == 4: self.cursor_pos[0] += 1 # Right
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.BOARD_COLS - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.BOARD_ROWS - 1)

        if space_pressed:
            reward += self._action_place_fragment()
        
        if shift_pressed:
            reward += self._action_manipulate_dice()

        # Update score
        self.score += reward

        # Check termination conditions
        terminated = self.game_over or self.steps >= 1000
        if self.win_state:
            reward += 100
            self.score += 100
            self.puzzles_solved_session += 1
            self._set_feedback("PUZZLE COMPLETE!", self.COLOR_GOAL, 120)
            terminated = True
        elif self.game_over and not self.win_state: # Detected
            reward -= 100
            self.score -= 100
            self._set_feedback("DETECTION!", self.COLOR_DETECTION, 120)
            terminated = True
        elif self.steps >= 1000:
            self._set_feedback("TIME LIMIT REACHED", self.COLOR_DETECTION, 120)
            terminated = True
        
        self.game_over = terminated

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _action_place_fragment(self):
        r, c = self.cursor_pos[1], self.cursor_pos[0]
        if self.available_fragments > 0 and self.board[r][c] == 0:
            # Sfx: Place fragment sound
            self.available_fragments -= 1
            self.board[r][c] = 1
            self._end_turn_logic()
            
            if (r, c) in self.puzzle_goal:
                self._set_feedback("Correct placement", self.COLOR_CURSOR)
                self._create_particles(c, r, self.COLOR_CURSOR)
                if self._check_win_condition():
                    self.win_state = True
                return 1.0
            else:
                self._set_feedback("Incorrect placement", self.COLOR_DETECTION)
                self._create_particles(c, r, self.COLOR_DETECTION)
                return -0.1
        else:
            # Sfx: Error sound
            self._set_feedback("Cannot place here", self.COLOR_GOAL)
            return 0.0

    def _action_manipulate_dice(self):
        if self.dice_manipulations_left > 0:
            # Sfx: Power-up sound
            self.dice_manipulations_left -= 1
            self.detection_modifier = 0.25 # Significantly reduce detection chance
            self._set_feedback("Dice manipulated!", self.COLOR_FRAGMENT)
            self._create_particles(self.cursor_pos[0], self.cursor_pos[1], self.COLOR_FRAGMENT, 30)
            return 5.0
        else:
            # Sfx: Error sound
            self._set_feedback("No manipulations left", self.COLOR_GOAL)
            return 0.0

    def _end_turn_logic(self):
        """Called after a fragment is placed, advancing the game turn."""
        num_placed = sum(row.count(1) for row in self.board)
        self.detection_prob = 0.10 + 0.02 * num_placed
        
        self.dice_roll = self.np_random.integers(1, 7)
        self.dice_anim_timer = 15 # Animate for 15 frames

        # Check for detection on unfavorable rolls (e.g., 1 or 6)
        if self.dice_roll in [1, 6]:
            effective_prob = self.detection_prob * self.detection_modifier
            if self.np_random.random() < effective_prob:
                # Sfx: Detection alarm
                self.game_over = True
        
        self.detection_modifier = 1.0 # Reset modifier after use

    def _check_win_condition(self):
        for r, c in self.puzzle_goal:
            if self.board[r][c] == 0:
                return False
        return True

    def _generate_puzzle(self):
        self.puzzle_goal.clear()
        num_elements = min(self.BOARD_COLS * self.BOARD_ROWS - 1, 2 + self.level)
        
        start_r = self.np_random.integers(0, self.BOARD_ROWS)
        start_c = self.np_random.integers(0, self.BOARD_COLS)
        self.puzzle_goal.append((start_r, start_c))
        
        visited = set([(start_r, start_c)])
        
        for _ in range(num_elements - 1):
            # Find a random placed piece and expand from it
            base_r, base_c = self.np_random.choice(self.puzzle_goal)
            
            attempts = 0
            while attempts < 20:
                # Try to place a new piece adjacent to an existing one
                dr, dc = self.np_random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
                new_r, new_c = base_r + dr, base_c + dc
                
                if 0 <= new_r < self.BOARD_ROWS and 0 <= new_c < self.BOARD_COLS and (new_r, new_c) not in visited:
                    self.puzzle_goal.append((new_r, new_c))
                    visited.add((new_r, new_c))
                    break
                attempts += 1
            if attempts == 20: # Could not expand, stop early
                break

    def _get_observation(self):
        self._update_visuals()
        self.screen.fill(self.COLOR_BG)
        self._render_background_effects()
        self._render_board()
        self._render_particles()
        self._render_ui()
        if self.game_over:
            self._render_game_over_overlay()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "level": self.level}

    def _update_visuals(self):
        """Update animations and timers."""
        # Interpolate cursor
        self.visual_cursor_pos[0] += (self.cursor_pos[0] - self.visual_cursor_pos[0]) * 0.4
        self.visual_cursor_pos[1] += (self.cursor_pos[1] - self.visual_cursor_pos[1]) * 0.4

        # Update particles
        self.particles = [p for p in self.particles if p.lifetime > 0]
        for p in self.particles:
            p.update()
        
        if self.feedback_timer > 0:
            self.feedback_timer -= 1
        
        if self.dice_anim_timer > 0:
            self.dice_anim_timer -= 1

    def _render_background_effects(self):
        """Draws subtle, slow-moving stars for a dreamlike feel."""
        for i in range(40):
            x = (hash(i * 10) + self.steps / 5) % self.WIDTH
            y = (hash(i * 33) + self.steps / 8) % self.HEIGHT
            alpha = 50 + math.sin(self.steps / 30 + i) * 20
            pygame.gfxdraw.pixel(self.screen, int(x), int(y), (*self.COLOR_GRID, int(alpha)))

    def _render_board(self):
        # Draw grid
        for r in range(self.BOARD_ROWS + 1):
            y = self.BOARD_OFFSET_Y + r * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.BOARD_OFFSET_X, y), (self.BOARD_OFFSET_X + self.GRID_WIDTH, y), 1)
        for c in range(self.BOARD_COLS + 1):
            x = self.BOARD_OFFSET_X + c * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.BOARD_OFFSET_Y), (x, self.BOARD_OFFSET_Y + self.GRID_HEIGHT), 1)
        
        # Draw goal locations
        for r, c in self.puzzle_goal:
            if self.board[r][c] == 0:
                rect = pygame.Rect(self.BOARD_OFFSET_X + c * self.CELL_SIZE, self.BOARD_OFFSET_Y + r * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
                pygame.draw.rect(self.screen, self.COLOR_GOAL, rect, 2, border_radius=4)
                
        # Draw placed fragments
        for r in range(self.BOARD_ROWS):
            for c in range(self.BOARD_COLS):
                if self.board[r][c] == 1:
                    center_x = self.BOARD_OFFSET_X + c * self.CELL_SIZE + self.CELL_SIZE // 2
                    center_y = self.BOARD_OFFSET_Y + r * self.CELL_SIZE + self.CELL_SIZE // 2
                    self._draw_glowing_circle(self.screen, self.COLOR_FRAGMENT, (center_x, center_y), self.CELL_SIZE // 3, 1.8)

        # Draw cursor
        vis_c, vis_r = self.visual_cursor_pos
        cursor_rect = pygame.Rect(
            self.BOARD_OFFSET_X + vis_c * self.CELL_SIZE,
            self.BOARD_OFFSET_Y + vis_r * self.CELL_SIZE,
            self.CELL_SIZE, self.CELL_SIZE
        )
        # Create a transparent surface for the cursor
        surf = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
        pygame.draw.rect(surf, (*self.COLOR_CURSOR, 100), surf.get_rect(), border_radius=4)
        pygame.draw.rect(surf, (*self.COLOR_CURSOR, 255), surf.get_rect(), 2, border_radius=4)
        self.screen.blit(surf, cursor_rect.topleft)

    def _render_ui(self):
        ui_x = self.BOARD_OFFSET_X + self.GRID_WIDTH + 20
        
        # --- Stats ---
        self._draw_text(f"Score: {self.score}", (ui_x, 40), self.COLOR_TEXT, self.font_medium)
        self._draw_text(f"Fragments: {self.available_fragments}", (ui_x, 70), self.COLOR_TEXT, self.font_small)
        self._draw_text(f"Manipulate: {self.dice_manipulations_left}", (ui_x, 90), self.COLOR_TEXT, self.font_small)
        self._draw_text(f"Level: {self.level}", (ui_x, 110), self.COLOR_TEXT, self.font_small)

        # --- Detection Info ---
        self._draw_text("Subconscious", (ui_x, 160), self.COLOR_DETECTION, self.font_medium)
        self._draw_text(f"Detect Prob: {self.detection_prob:.0%}", (ui_x, 190), self.COLOR_TEXT, self.font_small)
        self._render_dice((ui_x + 45, 250))
        
        # --- Feedback Text ---
        if self.feedback_timer > 0:
            alpha = min(255, int(255 * (self.feedback_timer / 30.0)))
            color = (*self.feedback_color, alpha)
            self._draw_text(self.feedback_text, (self.WIDTH / 2, self.HEIGHT - 20), color, self.font_medium, center=True)

    def _render_dice(self, center):
        size = 50
        rect = pygame.Rect(center[0] - size//2, center[1] - size//2, size, size)
        
        roll = self.dice_roll
        if self.dice_anim_timer > 0:
            roll = self.np_random.integers(1, 7)
            
        pygame.draw.rect(self.screen, self.COLOR_DICE, rect, 0, border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_BG, rect, 2, border_radius=5)

        dot_radius = 4
        positions = {
            1: [(0.5, 0.5)],
            2: [(0.25, 0.25), (0.75, 0.75)],
            3: [(0.25, 0.25), (0.5, 0.5), (0.75, 0.75)],
            4: [(0.25, 0.25), (0.25, 0.75), (0.75, 0.25), (0.75, 0.75)],
            5: [(0.25, 0.25), (0.25, 0.75), (0.5, 0.5), (0.75, 0.25), (0.75, 0.75)],
            6: [(0.25, 0.25), (0.25, 0.5), (0.25, 0.75), (0.75, 0.25), (0.75, 0.5), (0.75, 0.75)]
        }
        for dx, dy in positions.get(roll, []):
            pygame.draw.circle(self.screen, self.COLOR_BG, (rect.x + dx * size, rect.y + dy * size), dot_radius)

    def _render_particles(self):
        for p in self.particles:
            p.draw(self.screen)

    def _render_game_over_overlay(self):
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        color = self.COLOR_DETECTION if not self.win_state else self.COLOR_GOAL
        overlay.fill((*color, 100))
        self.screen.blit(overlay, (0, 0))

        text = "PUZZLE COMPLETE" if self.win_state else "DETECTED"
        text_surf = self.font_large.render(text, True, self.COLOR_TEXT)
        text_rect = text_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
        self.screen.blit(text_surf, text_rect)

    def _draw_glowing_circle(self, surface, color, center, radius, glow_factor):
        glow_radius = int(radius * glow_factor)
        
        # Create a temporary surface for the glow effect
        temp_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        
        # Draw multiple circles with decreasing alpha for a smooth glow
        for i in range(glow_radius - radius, 0, -2):
            alpha = int(80 * (1 - (i / (glow_radius - radius))))
            pygame.gfxdraw.filled_circle(temp_surf, glow_radius, glow_radius, radius + i, (*color, alpha))
        
        # Draw the main circle
        pygame.gfxdraw.filled_circle(temp_surf, glow_radius, glow_radius, radius, color)
        pygame.gfxdraw.aacircle(temp_surf, glow_radius, glow_radius, radius, color)
        
        surface.blit(temp_surf, (int(center[0] - glow_radius), int(center[1] - glow_radius)))

    def _draw_text(self, text, pos, color, font, center=False):
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        if center:
            text_rect.center = (int(pos[0]), int(pos[1]))
        else:
            text_rect.topleft = (int(pos[0]), int(pos[1]))
        self.screen.blit(text_surface, text_rect)

    def _create_particles(self, c, r, color, count=20):
        center_x = self.BOARD_OFFSET_X + c * self.CELL_SIZE + self.CELL_SIZE // 2
        center_y = self.BOARD_OFFSET_Y + r * self.CELL_SIZE + self.CELL_SIZE // 2
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 3)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            lifetime = random.randint(20, 40)
            radius = random.randint(2, 5)
            self.particles.append(Particle(center_x, center_y, vx, vy, color, radius, lifetime))

    def _set_feedback(self, text, color, duration=60):
        self.feedback_text = text
        self.feedback_color = color
        self.feedback_timer = duration

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # Example of how to run the environment
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play Example ---
    obs, info = env.reset()
    done = False
    
    # Use a display window for manual play
    # This requires a non-dummy SDL_VIDEODRIVER
    os.environ["SDL_VIDEODRIVER"] = "x11" # or "windows", "macOS"
    display_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Dream Logic Circuit")
    
    action = [0, 0, 0] # no-op, released, released
    
    while not done:
        # Map keyboard keys to actions for manual testing
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            # Keydown events for press-once actions
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    action[1] = 1
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                    action[2] = 1
        
        # Check held keys for continuous actions
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        else: action[0] = 0
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Reset one-shot actions after the step
        action[1] = 0
        action[2] = 0
        
        env.clock.tick(30) # Limit to 30 FPS

    # Keep the final screen visible for a moment
    pygame.time.wait(2000)
    env.close()