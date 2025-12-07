import gymnasium as gym
import os
import pygame
import numpy as np
import pygame.gfxdraw
import math
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:47:07.132104
# Source Brief: brief_01308.md
# Brief Index: 1308
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Match the color sequence by placing shapes on the grid. Camouflage your shapes "
        "by matching the background color to avoid detection by the sweeping beam."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the cursor. Press space to place a shape "
        "and shift to rotate it."
    )
    auto_advance = True

    # --- CONSTANTS ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_WIDTH, GRID_HEIGHT = 16, 10
    CELL_SIZE = 40
    MAX_STEPS = 2500
    NUM_SEQUENCES_TO_WIN = 5
    MAX_DETECTIONS = 3

    # Colors
    COLOR_BG = (15, 18, 28)
    COLOR_GRID = (30, 35, 50)
    COLOR_CURSOR = (255, 255, 255)
    COLOR_TEXT = (220, 220, 230)
    COLOR_TEXT_GLOW = (100, 100, 255)
    COLOR_BEAM = (255, 220, 50)
    COLOR_DETECTED = (255, 50, 80)

    TETRA_COLORS = [
        (50, 200, 255),  # Cyan
        (255, 100, 200),  # Pink
        (100, 255, 150),  # Mint
        (255, 180, 50),  # Orange
        (200, 150, 255),  # Lavender
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 20)
        self.font_large = pygame.font.Font(None, 32)

        # State variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.cursor_pos = [0, 0]
        self.visual_cursor_pos = [0.0, 0.0]
        self.active_tetra = None
        self.placed_tetras = []
        self.rhythm_sequence = []
        self.sequence_progress = 0
        self.completed_sequences = 0
        self.detections = 0
        self.beam_y = 0.0
        self.beam_speed = 0.0
        self.background_pattern = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=int)
        self.particles = []
        self.prev_space_held = False
        self.prev_shift_held = False
        self.last_reward_event = ""
        self.reward_event_timer = 0
        self.available_colors = 3
        self.tetra_pulse_rate = 0.0
        self.bg_change_timer = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False

        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.visual_cursor_pos = [
            self.cursor_pos[0] * self.CELL_SIZE + self.CELL_SIZE / 2,
            self.cursor_pos[1] * self.CELL_SIZE + self.CELL_SIZE / 2
        ]

        self.placed_tetras = []
        self.sequence_progress = 0
        self.completed_sequences = 0
        self.detections = 0
        self.beam_y = -20
        self.particles = []
        self.prev_space_held = False
        self.prev_shift_held = False
        self.last_reward_event = ""
        self.reward_event_timer = 0

        self._update_difficulty()
        self._generate_background_pattern()
        self._generate_new_sequence()
        self._generate_new_active_tetra()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # --- Handle Input ---
        reward += self._handle_input(movement, space_held, shift_held)

        # --- Update Game State ---
        self._update_game_state()

        # --- Calculate Rewards & Check Detections ---
        reward += self._process_rewards_and_detections()

        self.score += reward

        # --- Check Termination ---
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        if terminated or truncated:
            self.game_over = True
            if self.completed_sequences >= self.NUM_SEQUENCES_TO_WIN:
                reward += 50
                self._add_reward_event("VICTORY!", 50)
            elif self.detections >= self.MAX_DETECTIONS:
                self._add_reward_event("FAILURE", -50)
                # The -50 for detection is applied when it happens, no need to add again
        
        terminated = self.game_over

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, movement, space_held, shift_held):
        reward = 0
        # Rotation (Shift)
        if shift_held and not self.prev_shift_held and self.active_tetra:
            self.active_tetra['rotation'] = (self.active_tetra['rotation'] + 90) % 360
            # sfx: rotate_sound

        # Movement
        if movement != 0 and self.active_tetra:
            new_pos = self.cursor_pos[:]
            if movement == 1: new_pos[1] -= 1  # Up
            elif movement == 2: new_pos[1] += 1  # Down
            elif movement == 3: new_pos[0] -= 1  # Left
            elif movement == 4: new_pos[0] += 1  # Right

            if 0 <= new_pos[0] < self.GRID_WIDTH and 0 <= new_pos[1] < self.GRID_HEIGHT:
                self.cursor_pos = new_pos

        # Placement (Space)
        if space_held and not self.prev_space_held and self.active_tetra:
            if not any(p['grid_pos'] == self.cursor_pos for p in self.placed_tetras):
                place_reward = self._place_tetra()
                reward += place_reward
        return reward

    def _place_tetra(self):
        # Rhythm Check (size pulse)
        pulse_phase = (self.steps * self.tetra_pulse_rate) % (2 * math.pi)
        rhythm_bonus = max(0, math.sin(pulse_phase))  # 0 to 1
        is_good_rhythm = rhythm_bonus > 0.9

        # Color Check
        is_correct_color = self.active_tetra['color_idx'] == self.rhythm_sequence[self.sequence_progress]

        # Camouflage Check
        bg_color_idx = self.background_pattern[self.cursor_pos[0], self.cursor_pos[1]]
        is_camouflaged = self.active_tetra['color_idx'] == bg_color_idx

        placed_tetra = {
            'grid_pos': self.cursor_pos[:],
            'color_idx': self.active_tetra['color_idx'],
            'rotation': self.active_tetra['rotation'],
            'is_camouflaged': is_camouflaged,
            'is_correct_sequence': is_correct_color,
            'is_detected': False,
            'spawn_step': self.steps
        }
        self.placed_tetras.append(placed_tetra)

        if is_good_rhythm and is_correct_color:
            self.sequence_progress += 1
            # sfx: place_success
            self._spawn_particles(self.visual_cursor_pos, self.TETRA_COLORS[placed_tetra['color_idx']], 20, 3)
            self._add_reward_event("Rhythm Hit!", 1)

            if self.sequence_progress >= len(self.rhythm_sequence):
                self.completed_sequences += 1
                self.sequence_progress = 0
                self.placed_tetras.clear()
                self._update_difficulty()
                self._generate_new_sequence()
                # sfx: sequence_complete
                self._spawn_particles((self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2), (255, 255, 100), 100, 5)
                self._add_reward_event("Sequence Complete!", 5)
                self._generate_background_pattern()
                self._generate_new_active_tetra()
                return 6  # 1 for placement + 5 for sequence

            self._generate_new_active_tetra()
            return 1
        else:
            # sfx: place_fail
            self._spawn_particles(self.visual_cursor_pos, self.COLOR_DETECTED, 10, 2)
            self._generate_new_active_tetra()
            return -0.5  # Small penalty for incorrect placement

    def _update_game_state(self):
        # Update visual cursor position (lerp)
        target_x = self.cursor_pos[0] * self.CELL_SIZE + self.CELL_SIZE / 2
        target_y = self.cursor_pos[1] * self.CELL_SIZE + self.CELL_SIZE / 2
        self.visual_cursor_pos[0] += (target_x - self.visual_cursor_pos[0]) * 0.4
        self.visual_cursor_pos[1] += (target_y - self.visual_cursor_pos[1]) * 0.4

        # Update beam
        self.beam_y += self.beam_speed
        if self.beam_y > self.SCREEN_HEIGHT + 20:
            self.beam_y = -20
            # Reset detection status for all tetras for the next sweep
            for tetra in self.placed_tetras:
                tetra['is_detected'] = False

        # Update particles
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.05  # gravity
            p['life'] -= 1

        # Update timers
        if self.reward_event_timer > 0:
            self.reward_event_timer -= 1

        self.bg_change_timer -= 1
        if self.bg_change_timer <= 0:
            self._generate_background_pattern()
            self.bg_change_timer = 200 - self.completed_sequences * 10

    def _process_rewards_and_detections(self):
        reward = 0
        camouflaged_count = 0

        beam_grid_y = int(self.beam_y / self.CELL_SIZE)

        for tetra in self.placed_tetras:
            if tetra['is_camouflaged']:
                camouflaged_count += 1
            elif not tetra['is_detected'] and tetra['grid_pos'][1] == beam_grid_y:
                self.detections += 1
                tetra['is_detected'] = True
                reward -= 50
                # sfx: detection_alarm
                self._spawn_particles(
                    (tetra['grid_pos'][0] * self.CELL_SIZE + self.CELL_SIZE / 2,
                     tetra['grid_pos'][1] * self.CELL_SIZE + self.CELL_SIZE / 2),
                    self.COLOR_DETECTED, 50, 7)
                self._add_reward_event("DETECTION!", -50)

        # Continuous reward for being camouflaged
        reward += camouflaged_count * 0.01
        return reward

    def _check_termination(self):
        return (
                self.detections >= self.MAX_DETECTIONS or
                self.completed_sequences >= self.NUM_SEQUENCES_TO_WIN
        )

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
            "detections": self.detections,
            "completed_sequences": self.completed_sequences
        }

    def _render_game(self):
        self._render_background()
        self._render_placed_tetras()
        if self.active_tetra:
            self._render_active_tetra()
        self._render_particles()
        self._render_beam()

    def _render_background(self):
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                color_idx = self.background_pattern[x, y]
                base_color = self.TETRA_COLORS[color_idx]
                # Desaturate and darken for background
                bg_color = tuple(int(c * 0.3) for c in base_color)
                rect = pygame.Rect(x * self.CELL_SIZE, y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
                pygame.draw.rect(self.screen, bg_color, rect)
        # Grid lines
        for x in range(self.GRID_WIDTH + 1):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x * self.CELL_SIZE, 0),
                             (x * self.CELL_SIZE, self.SCREEN_HEIGHT))
        for y in range(self.GRID_HEIGHT + 1):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y * self.CELL_SIZE),
                             (self.SCREEN_WIDTH, y * self.CELL_SIZE))

    def _render_placed_tetras(self):
        for tetra in self.placed_tetras:
            pos = (
                tetra['grid_pos'][0] * self.CELL_SIZE + self.CELL_SIZE / 2,
                tetra['grid_pos'][1] * self.CELL_SIZE + self.CELL_SIZE / 2
            )
            color = self.TETRA_COLORS[tetra['color_idx']]
            if not tetra['is_camouflaged']:
                # Make it stand out if not camouflaged
                pulse = abs(math.sin(self.steps * 0.1))
                color = (
                    min(255, color[0] + int(pulse * 50)),
                    max(0, color[1] - int(pulse * 50)),
                    max(0, color[2] - int(pulse * 50))
                )

            age = self.steps - tetra['spawn_step']
            alpha = min(255, 50 + age * 2)

            self._draw_tetra(self.screen, pos, tetra['rotation'], self.CELL_SIZE * 0.35, color, alpha)

    def _render_active_tetra(self):
        # Cursor
        size = self.CELL_SIZE * (0.6 + 0.1 * math.sin(self.steps * 0.2))
        pygame.gfxdraw.aacircle(self.screen, int(self.visual_cursor_pos[0]), int(self.visual_cursor_pos[1]),
                                int(size / 2), self.COLOR_CURSOR)

        # Pulsing tetra
        pulse_phase = (self.steps * self.tetra_pulse_rate) % (2 * math.pi)
        size_mod = 0.5 + 0.5 * math.sin(pulse_phase)  # 0 to 1
        tetra_size = self.CELL_SIZE * 0.3 * (0.8 + 0.4 * size_mod)
        color = self.TETRA_COLORS[self.active_tetra['color_idx']]

        self._draw_tetra(self.screen, self.visual_cursor_pos, self.active_tetra['rotation'], tetra_size, color, 255)

    def _render_beam(self):
        beam_h = 15
        beam_rect = pygame.Rect(0, self.beam_y - beam_h / 2, self.SCREEN_WIDTH, beam_h)

        # Create a surface for transparency
        s = pygame.Surface((self.SCREEN_WIDTH, beam_h), pygame.SRCALPHA)

        # Main beam color
        pygame.draw.rect(s, self.COLOR_BEAM + (80,), s.get_rect())

        # Core line
        pygame.draw.line(s, self.COLOR_BEAM + (200,), (0, beam_h // 2), (self.SCREEN_WIDTH, beam_h // 2), 2)

        self.screen.blit(s, (0, self.beam_y - beam_h / 2))

    def _render_particles(self):
        for p in self.particles:
            size = p['life'] * 0.1 * p['start_size']
            if size > 1:
                alpha = max(0, min(255, p['life'] * 10))
                color = p['color'] + (int(alpha),)
                s = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
                pygame.draw.circle(s, color, (int(size), int(size)), int(size))
                self.screen.blit(s, (int(p['pos'][0] - size), int(p['pos'][1] - size)))

    def _render_ui(self):
        # Score and Steps
        self._draw_glow_text(f"Score: {int(self.score)}", (10, 10), self.font_small)
        self._draw_glow_text(f"Steps: {self.steps}/{self.MAX_STEPS}", (self.SCREEN_WIDTH - 120, 10), self.font_small)

        # Detections
        for i in range(self.MAX_DETECTIONS):
            pos = (self.SCREEN_WIDTH - 30 - i * 25, self.SCREEN_HEIGHT - 25)
            if i < self.detections:
                color = self.COLOR_DETECTED
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 8, color)
            else:
                color = self.COLOR_TEXT
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 8, color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 9, color)

        # Rhythm Sequence
        seq_x_start = 20
        for i in range(len(self.rhythm_sequence)):
            pos = (seq_x_start + i * 40, self.SCREEN_HEIGHT - 25)
            color_idx = self.rhythm_sequence[i]
            color = self.TETRA_COLORS[color_idx]
            if i < self.sequence_progress:
                # Completed part of sequence
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 10, color)
            elif i == self.sequence_progress:
                # Current target
                pulse = 10 + 3 * abs(math.sin(self.steps * 0.2))
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(pulse), color)
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], int(pulse), color)
            else:
                # Future part
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 10, tuple(int(c * 0.5) for c in color))

        # Reward event text
        if self.reward_event_timer > 0:
            alpha = min(255, self.reward_event_timer * 5)
            color = self.COLOR_TEXT_GLOW if self.last_reward_amount > 0 else self.COLOR_DETECTED
            self._draw_glow_text(self.last_reward_event, (self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 - 50),
                                 self.font_large, color, alpha)

    def _draw_tetra(self, surface, center_pos, angle, size, color, alpha):
        points = []
        rad_angle = math.radians(angle)
        for i in range(3):
            a = rad_angle + (i * 2 * math.pi / 3)
            x = center_pos[0] + size * math.cos(a)
            y = center_pos[1] + size * math.sin(a)
            points.append((int(x), int(y)))

        center_point = (int(center_pos[0]), int(center_pos[1]))

        # For glow effect
        if alpha == 255:
            for i in range(5, 0, -1):
                glow_alpha = 40 - i * 5
                glow_color = color + (glow_alpha,)
                s = pygame.Surface((size * 2.5, size * 2.5), pygame.SRCALPHA)
                glow_points = [(p[0] - center_pos[0] + size * 1.25, p[1] - center_pos[1] + size * 1.25) for p in
                               points]
                pygame.gfxdraw.filled_trigon(s, int(glow_points[0][0]), int(glow_points[0][1]),
                                             int(glow_points[1][0]), int(glow_points[1][1]), int(glow_points[2][0]),
                                             int(glow_points[2][1]), glow_color)
                surface.blit(s, (center_pos[0] - size * 1.25, center_pos[1] - size * 1.25))

        # Main shape
        final_color = color + (alpha,) if alpha < 255 else color
        pygame.gfxdraw.filled_trigon(surface, points[0][0], points[0][1], points[1][0], points[1][1], points[2][0],
                                     points[2][1], final_color)
        pygame.gfxdraw.aatrigon(surface, points[0][0], points[0][1], points[1][0], points[1][1], points[2][0],
                                points[2][1], final_color)

        # Internal lines for 3D feel
        line_color = (255, 255, 255, int(alpha * 0.7))
        for p in points:
            pygame.draw.aaline(surface, line_color, p, center_point)

    def _draw_glow_text(self, text, pos, font, color=COLOR_TEXT, alpha=255):
        text_surf = font.render(text, True, color)
        glow_surf = font.render(text, True, self.COLOR_TEXT_GLOW)

        text_surf.set_alpha(alpha)
        glow_surf.set_alpha(int(alpha * 0.5))

        # Center align text
        text_rect = text_surf.get_rect(center=pos)
        glow_rect = glow_surf.get_rect(center=pos)

        # Blit glow with offsets
        self.screen.blit(glow_surf, glow_rect.move(1, 1))
        self.screen.blit(glow_surf, glow_rect.move(-1, -1))
        self.screen.blit(glow_surf, glow_rect.move(1, -1))
        self.screen.blit(glow_surf, glow_rect.move(-1, 1))

        # Blit main text
        self.screen.blit(text_surf, text_rect)

    def _generate_new_sequence(self):
        seq_len = 3 + self.completed_sequences
        self.rhythm_sequence = [self.np_random.integers(0, self.available_colors) for _ in range(seq_len)]

    def _generate_new_active_tetra(self):
        if self.sequence_progress >= len(self.rhythm_sequence):
            self.active_tetra = None
            return

        color_idx = self.rhythm_sequence[self.sequence_progress]
        self.active_tetra = {
            'color_idx': color_idx,
            'rotation': self.np_random.integers(0, 4) * 90,
        }

    def _generate_background_pattern(self):
        self.background_pattern = self.np_random.integers(0, self.available_colors,
                                                          size=(self.GRID_WIDTH, self.GRID_HEIGHT))

        # Anti-softlock: ensure required colors are available
        if self.sequence_progress < len(self.rhythm_sequence):
            required_color = self.rhythm_sequence[self.sequence_progress]
            if not np.any(self.background_pattern == required_color):
                # Place at least one instance of the required color
                num_placements = max(1, self.GRID_WIDTH * self.GRID_HEIGHT // 20)
                for _ in range(num_placements):
                    x, y = self.np_random.integers(0, self.GRID_WIDTH), self.np_random.integers(0, self.GRID_HEIGHT)
                    self.background_pattern[x, y] = required_color

    def _update_difficulty(self):
        level = self.completed_sequences
        self.beam_speed = 0.8 + level * 0.15
        self.tetra_pulse_rate = 0.08 + level * 0.01
        self.available_colors = min(len(self.TETRA_COLORS), 3 + level // 2)

    def _spawn_particles(self, pos, color, count, max_speed):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, max_speed)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': self.np_random.integers(30, 60),
                'color': color,
                'start_size': self.np_random.uniform(1.0, 1.5)
            })

    def _add_reward_event(self, text, amount):
        self.last_reward_event = text
        self.last_reward_amount = amount
        self.reward_event_timer = 60

    def close(self):
        pygame.quit()
        super().close()


if __name__ == '__main__':
    # This block allows you to play the game manually
    # Make sure to unset the dummy video driver if you want to see the game
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()

    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Rhythm Camouflage")
    clock = pygame.time.Clock()

    running = True
    total_reward = 0

    while running:
        movement, space, shift = 0, 0, 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4

        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

        action = [movement, space, shift]

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Transpose for pygame display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            print(f"Sequences Completed: {info['completed_sequences']}")
            obs, info = env.reset()
            total_reward = 0

        clock.tick(30)  # Run at 30 FPS

    env.close()