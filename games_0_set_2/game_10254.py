import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:10:34.739244
# Source Brief: brief_00254.md
# Brief Index: 254
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Escape a procedurally generated nightmare by matching musical notes to lull
    monsters to sleep and strategically building escape routes.

    Action Space: MultiDiscrete([5, 2, 2])
    - actions[0]: Movement (0=none, 1=up, 2=down, 3=left, 4=right)
    - actions[1]: Space button (0=released, 1=held)
    - actions[2]: Shift button (0=released, 1=held) -> Toggles mode

    Modes:
    - Build Mode (Shift Released): Movement moves a cursor, Space builds a path.
    - Note Mode (Shift Held): Movement matches notes to keep monsters asleep.
    """
    metadata = {"render_modes": ["rgb_array"]}
    game_description = (
        "Escape a procedurally generated nightmare by building a path to the exit. "
        "Keep monsters asleep by matching sequences of musical notes."
    )
    user_guide = (
        "Controls: Use arrow keys to move the cursor (Build Mode) or match notes (Note Mode). "
        "Press space to build the path. Hold shift to switch to Note Mode."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # === Gymnasium Interface ===
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # === Game Constants ===
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 2000
        self.GRID_SIZE = 40
        self.GRID_W = self.WIDTH // self.GRID_SIZE
        self.GRID_H = self.HEIGHT // self.GRID_SIZE

        # === Visual Design ===
        self.COLOR_BG = (15, 10, 35)
        self.COLOR_GRID = (30, 20, 60)
        self.COLOR_START = (0, 255, 150)
        self.COLOR_EXIT = (255, 0, 150)
        self.COLOR_ROUTE = (0, 200, 255)
        self.COLOR_CURSOR = (255, 255, 0)
        self.COLOR_MONSTER_SLEEP = (70, 70, 200)
        self.COLOR_MONSTER_AWAKE = (255, 50, 50)
        self.COLOR_TIMER_BAR = (200, 200, 255)
        self.COLOR_TEXT = (220, 220, 240)
        self.NOTE_COLORS = [
            (255, 100, 100), # Up
            (100, 255, 100), # Down
            (100, 100, 255), # Left
            (255, 255, 100)  # Right
        ]

        # === Pygame Setup ===
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 32, bold=True)

        # === State Variables (initialized in reset) ===
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.start_pos = (0, 0)
        self.exit_pos = (0, 0)
        self.cursor_pos = (0, 0)
        self.route_nodes = []
        self.monsters = []
        self.notes = []
        self.particles = []
        self.is_note_mode = False
        self.max_notes = 1
        self.base_sleep_duration = 5.0 # seconds
        self.last_cursor_move_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.is_note_mode = False
        self.particles = []
        self.last_cursor_move_step = 0

        # Difficulty scaling reset
        self.max_notes = 1
        self.base_sleep_duration = 5.0 * self.FPS

        # World generation
        self.start_pos = (1, self.GRID_H // 2)
        self.exit_pos = (self.GRID_W - 2, self.GRID_H // 2)
        self.cursor_pos = self.start_pos
        self.route_nodes = [self.start_pos]

        # Monster generation
        self.monsters = []
        num_monsters = 3
        for i in range(num_monsters):
            while True:
                px1 = self.np_random.integers(3, self.GRID_W - 3)
                py1 = self.np_random.integers(1, self.GRID_H - 1)
                px2 = self.np_random.integers(3, self.GRID_W - 3)
                py2 = self.np_random.integers(1, self.GRID_H - 1)
                if math.dist((px1, py1), (px2, py2)) > 4:
                    break
            self.monsters.append({
                "p1": (px1 * self.GRID_SIZE, py1 * self.GRID_SIZE),
                "p2": (px2 * self.GRID_SIZE, py2 * self.GRID_SIZE),
                "pos": np.array(self.to_pixel(px1, py1), dtype=float),
                "t": 0.0,
                "speed": self.np_random.uniform(0.01, 0.02),
                "is_sleeping": True,
                "sleep_timer": self.base_sleep_duration,
            })

        self._generate_notes()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0.0
        self.steps += 1

        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        shift_held = action[2] == 1

        self.is_note_mode = shift_held

        # --- Action Handling ---
        if self.is_note_mode:
            # Note Matching Logic
            if movement in [1, 2, 3, 4]:
                note_idx_to_match = movement - 1
                if note_idx_to_match < len(self.notes) and self.notes[0] == note_idx_to_match:
                    matched_note = self.notes.pop(0)
                    reward += 1
                    self.score += 1
                    self._create_particles(self._get_note_pos(0), self.NOTE_COLORS[matched_note], 20)
                    # // Sound: Note Match Success
                    if not self.notes:
                        # All notes cleared, reset monster timers
                        for m in self.monsters:
                            m["sleep_timer"] = self.base_sleep_duration
                        self._generate_notes()
                        # // Sound: Note Sequence Complete
                elif note_idx_to_match < len(self.notes): # Wrong note
                    reward -= 2 # Penalty for wrong note
                    self.game_over = True # Harsh penalty for mistake

        else:
            # Build Mode Logic
            if movement != 0 and self.steps > self.last_cursor_move_step: # Debounce cursor movement
                self.last_cursor_move_step = self.steps
                dx, dy = [(0,0), (0,-1), (0,1), (-1,0), (1,0)][movement]
                new_cursor_pos = (self.cursor_pos[0] + dx, self.cursor_pos[1] + dy)
                if 0 <= new_cursor_pos[0] < self.GRID_W and 0 <= new_cursor_pos[1] < self.GRID_H:
                    self.cursor_pos = new_cursor_pos

            if space_held:
                last_node = self.route_nodes[-1]
                if self.cursor_pos not in self.route_nodes and math.dist(self.cursor_pos, last_node) < 1.5:
                    self.route_nodes.append(self.cursor_pos)
                    reward += 5
                    px, py = self.to_pixel(*self.cursor_pos)
                    self._create_particles((px, py), self.COLOR_ROUTE, 15)
                    # // Sound: Build Segment
                    if self.cursor_pos == self.exit_pos:
                        self.game_over = True
                        reward += 100
                        # // Sound: Victory

        # --- Game Logic Update ---
        self._update_difficulty()
        self._update_monsters()
        self._update_particles()

        # Monster state reward/punishment
        for m in self.monsters:
            if not m["is_sleeping"]:
                dist_to_path = min([math.dist(m["pos"], self.to_pixel(*node)) for node in self.route_nodes])
                if dist_to_path < self.GRID_SIZE * 0.8:
                    self.game_over = True
                    reward -= 100
                    # // Sound: Game Over / Caught

                # Punish for monster being close
                punishment = 1 / max(1, dist_to_path / self.GRID_SIZE)
                reward -= punishment * 0.1

        # Termination conditions
        terminated = self.game_over
        truncated = self.steps >= self.MAX_STEPS
        if truncated and not terminated:
            # Ran out of time, small penalty
            reward -= 10

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _update_difficulty(self):
        # Monster sleep duration decreases every 200 steps
        if self.steps > 0 and self.steps % 200 == 0:
            self.base_sleep_duration = max(2 * self.FPS, self.base_sleep_duration - (0.1 * self.FPS))

        # Number of notes increases every 500 steps
        if self.steps > 0 and self.steps % 500 == 0:
            self.max_notes = min(4, self.max_notes + 1)

    def _update_monsters(self):
        for m in self.monsters:
            if m["is_sleeping"]:
                m["sleep_timer"] -= 1
                if m["sleep_timer"] <= 0:
                    m["is_sleeping"] = False
                    m["sleep_timer"] = 0
                    px, py = m["pos"]
                    self._create_particles((px, py), self.COLOR_MONSTER_AWAKE, 30, life=40)
                    # // Sound: Monster Wake Alert
            else: # Awake
                # Target the closest point on the path
                path_pixels = [self.to_pixel(*node) for node in self.route_nodes]
                target = min(path_pixels, key=lambda p: math.dist(m["pos"], p))

                direction = np.array(target) - m["pos"]
                dist = np.linalg.norm(direction)
                if dist > 1:
                    m["pos"] += (direction / dist) * 2.0 # Awake speed

            # Patrol movement if sleeping
            if m["is_sleeping"]:
                m["t"] += m["speed"]
                if m["t"] > 1.0:
                    m["t"] = 1.0
                    m["p1"], m["p2"] = m["p2"], m["p1"] # Swap patrol points
                    m["t"] = 0.0
                m["pos"] = self._lerp_vec(m["p1"], m["p2"], m["t"])

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "mode": "note" if self.is_note_mode else "build",
            "notes_to_match": len(self.notes),
            "route_length": len(self.route_nodes),
        }

    def _render_game(self):
        self._draw_grid()
        self._draw_particles()
        self._draw_start_exit()
        self._draw_route()
        self._draw_monsters()
        if not self.is_note_mode:
            self._draw_cursor()
        if self.is_note_mode:
            self._draw_notes()

    def _draw_grid(self):
        for x in range(0, self.WIDTH, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

    def _draw_start_exit(self):
        start_px, start_py = self.to_pixel(*self.start_pos)
        exit_px, exit_py = self.to_pixel(*self.exit_pos)
        self._draw_glowing_circle(self.screen, self.COLOR_START, (start_px, start_py), 12, 3)
        self._draw_glowing_circle(self.screen, self.COLOR_EXIT, (exit_px, exit_py), 12, 3)

    def _draw_route(self):
        if len(self.route_nodes) > 1:
            points = [self.to_pixel(*node) for node in self.route_nodes]
            pygame.draw.lines(self.screen, self.COLOR_ROUTE, False, points, 5)
        for node in self.route_nodes:
            px, py = self.to_pixel(*node)
            self._draw_glowing_circle(self.screen, self.COLOR_ROUTE, (px, py), 4, 1)

    def _draw_cursor(self):
        px, py = self.to_pixel(*self.cursor_pos)
        rect = pygame.Rect(px - self.GRID_SIZE//2, py - self.GRID_SIZE//2, self.GRID_SIZE, self.GRID_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, rect, 2, border_radius=4)

    def _draw_monsters(self):
        for m in self.monsters:
            color = self.COLOR_MONSTER_SLEEP if m["is_sleeping"] else self.COLOR_MONSTER_AWAKE
            pos_int = (int(m["pos"][0]), int(m["pos"][1]))
            self._draw_glowing_circle(self.screen, color, pos_int, 10, 4)

            # Sleep timer bar
            if m["is_sleeping"]:
                bar_width = 25
                bar_height = 4
                bar_x = pos_int[0] - bar_width // 2
                bar_y = pos_int[1] - 20
                fill_ratio = m["sleep_timer"] / self.base_sleep_duration
                pygame.draw.rect(self.screen, self.COLOR_GRID, (bar_x, bar_y, bar_width, bar_height))
                pygame.draw.rect(self.screen, self.COLOR_TIMER_BAR, (bar_x, bar_y, int(bar_width * fill_ratio), bar_height))

    def _draw_notes(self):
        for i, note_idx in enumerate(self.notes):
            pos = self._get_note_pos(i)
            color = self.NOTE_COLORS[note_idx]
            self._draw_glowing_circle(self.screen, color, pos, 15, 3)

            # Draw direction indicator
            key_text = ["↑", "↓", "←", "→"][note_idx]
            text_surf = self.font_small.render(key_text, True, (0,0,0))
            text_rect = text_surf.get_rect(center=pos)
            self.screen.blit(text_surf, text_rect)

    def _render_ui(self):
        # Score
        score_surf = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (10, 10))

        # Steps
        steps_surf = self.font_small.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(steps_surf, (10, 45))

        # Mode Indicator
        mode_text = "NOTE MODE" if self.is_note_mode else "BUILD MODE"
        color = self.NOTE_COLORS[3] if self.is_note_mode else self.COLOR_CURSOR
        mode_surf = self.font_large.render(mode_text, True, color)
        mode_rect = mode_surf.get_rect(centerx=self.WIDTH // 2, y=10)
        self.screen.blit(mode_surf, mode_rect)

        if self.game_over:
            is_win = self.cursor_pos == self.exit_pos and not any(not m['is_sleeping'] for m in self.monsters)
            end_text = "ESCAPED!" if is_win else "CAUGHT!"
            end_color = self.COLOR_START if is_win else self.COLOR_MONSTER_AWAKE
            end_surf = self.font_large.render(end_text, True, end_color)
            end_rect = end_surf.get_rect(center=(self.WIDTH//2, self.HEIGHT//2))
            self.screen.blit(end_surf, end_rect)

    # --- Helper Functions ---
    def to_pixel(self, grid_x, grid_y):
        return (grid_x * self.GRID_SIZE + self.GRID_SIZE // 2,
                grid_y * self.GRID_SIZE + self.GRID_SIZE // 2)

    def _lerp_vec(self, v1, v2, t):
        return (v1[0] * (1 - t) + v2[0] * t, v1[1] * (1 - t) + v2[1] * t)

    def _generate_notes(self):
        self.notes = self.np_random.choice(4, self.max_notes, replace=False).tolist()

    def _get_note_pos(self, index):
        num_notes = len(self.notes)
        center_x, center_y = self.WIDTH // 2, 80
        total_width = (num_notes - 1) * 50
        start_x = center_x - total_width / 2
        return (int(start_x + index * 50), center_y)

    def _create_particles(self, pos, color, count, life=20):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({'pos': list(pos), 'vel': vel, 'life': life, 'color': color})

    def _draw_particles(self):
        for p in self.particles:
            alpha = max(0, 255 * (p['life'] / 20))
            # Create a temporary surface for alpha blending
            radius = int(p['life'] * 0.15)
            if radius <= 0: continue
            temp_surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, (*p['color'], alpha), (radius, radius), radius)
            pos_int = (int(p['pos'][0] - radius), int(p['pos'][1] - radius))
            self.screen.blit(temp_surf, pos_int)


    def _draw_glowing_circle(self, surface, color, center, radius, glow_strength):
        center_int = (int(center[0]), int(center[1]))
        for i in range(glow_strength, 0, -1):
            alpha = 80 - (i * 20)
            glow_color = (*color, alpha)
            temp_surf = pygame.Surface((radius*2 + i*4, radius*2 + i*4), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, glow_color, (radius+i*2, radius+i*2), radius + i*2)
            surface.blit(temp_surf, (center_int[0] - radius - i*2, center_int[1] - radius - i*2), special_flags=pygame.BLEND_RGBA_ADD)
        pygame.gfxdraw.aacircle(surface, center_int[0], center_int[1], radius, color)
        pygame.gfxdraw.filled_circle(surface, center_int[0], center_int[1], radius, color)

    def close(self):
        pygame.quit()

# Example usage:
if __name__ == "__main__":
    # To run with display, unset the dummy video driver
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv()
    obs, info = env.reset()

    # --- Manual Play Controls ---
    # Arrow Keys: Move cursor / Match notes
    # Space: Build path
    # Left Shift: Toggle Mode
    # R: Reset environment
    # Q: Quit

    running = True
    is_manual_mode = True

    if is_manual_mode:
        screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        pygame.display.set_caption("Dream Weaver")
        clock = pygame.time.Clock()

        action = np.array([0, 0, 0]) # no-op, released, released

        while running:
            # --- Event Handling ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q: running = False
                    if event.key == pygame.K_r: obs, info = env.reset()

            # --- Action Generation from Keystates ---
            keys = pygame.key.get_pressed()
            action[1] = 1 if keys[pygame.K_SPACE] else 0
            action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

            # Movement is discrete, so we only trigger it once per press
            # This is handled by checking the event queue
            move_action = 0
            for event in pygame.event.get(pygame.KEYDOWN):
                 if event.key == pygame.K_UP: move_action = 1
                 elif event.key == pygame.K_DOWN: move_action = 2
                 elif event.key == pygame.K_LEFT: move_action = 3
                 elif event.key == pygame.K_RIGHT: move_action = 4
            action[0] = move_action

            # --- Step the Environment ---
            # If auto_advance is True, we step every frame.
            # If it were False, we would only step on an action.
            obs, reward, terminated, truncated, info = env.step(action)

            # --- Render the observation to the display ---
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()

            if terminated or truncated:
                print(f"Game Over! Score: {info['score']}, Steps: {info['steps']}")
                # Wait a bit before resetting
                pygame.time.wait(2000)
                obs, info = env.reset()
                action.fill(0)

            clock.tick(env.FPS)

    env.close()