import gymnasium as gym
import os
import pygame
import pygame.gfxdraw
import math
import random
import numpy as np
from gymnasium.spaces import MultiDiscrete
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


def lerp(a, b, t):
    """Linear interpolation."""
    return a + (b - a) * t


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # FIX: Add game description
    game_description = (
        "Climb a mystical mountain by creating platforms. Match the musical harmony to build your path to the summit."
    )

    # FIX: Add user guide
    user_guide = (
        "Controls: ←→ to move, ↑ to jump. Press Shift to select notes for the harmony, and Space to create a platform."
    )

    # FIX: Add auto_advance flag
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.MAX_STEPS = 2000
        self.SUMMIT_HEIGHT = 1600  # Total height of the mountain world

        # Player physics
        self.PLAYER_SIZE = 12
        self.PLAYER_SPEED = 4.0
        self.PLAYER_JUMP_FORCE = 9.0
        self.GRAVITY = 0.4
        self.FRICTION = 0.85

        # Platform properties
        self.PLATFORM_HEIGHT = 10
        self.MAX_PLATFORMS = 4

        # Harmony/Puzzle properties
        self.NOTE_PALETTE = [
            (255, 64, 64),   # C (Red)
            (255, 165, 0),   # D (Orange)
            (255, 255, 0),   # E (Yellow)
            (0, 255, 0),     # F (Green)
            (0, 191, 255),   # G (Sky Blue)
            (75, 0, 130),    # A (Indigo)
            (238, 130, 238)  # B (Violet)
        ]
        self.MAX_HARMONY_LENGTH = 5

        # Rewards
        self.REWARD_WIN = 100.0
        self.REWARD_LOSS = -100.0
        self.REWARD_PLATFORM_SUCCESS = 5.0
        self.REWARD_PER_PIXEL_CLIMBED = 0.01

        # Colors
        self.COLOR_BG_TOP = (10, 20, 40)
        self.COLOR_BG_BOTTOM = (40, 20, 60)
        self.COLOR_PLAYER = (255, 255, 255)
        self.COLOR_PLAYER_GLOW = (200, 200, 255)
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.COLOR_FAILURE = (100, 100, 120)

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_harmony = pygame.font.SysFont("Verdana", 24, bold=True)

        # --- State Variables ---
        self.player_pos = None
        self.player_vel = None
        self.on_ground = None
        self.platforms = None
        self.particles = None
        self.camera_y = None
        self.highest_y = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.previous_space_held = None
        self.previous_shift_held = None
        self.target_harmony = None
        self.selected_notes = None
        self.harmony_difficulty = None
        self.successful_placements = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.player_pos = pygame.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 50)
        self.player_vel = pygame.Vector2(0, 0)
        self.on_ground = False

        start_platform_rect = pygame.Rect(
            self.SCREEN_WIDTH / 2 - 50, self.SCREEN_HEIGHT - 20, 100, self.PLATFORM_HEIGHT
        )
        self.platforms = [{"rect": start_platform_rect, "notes": []}]

        self.particles = []
        self.camera_y = 0
        self.highest_y = self.player_pos.y

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.previous_space_held = False
        self.previous_shift_held = False

        self.selected_notes = []
        self.harmony_difficulty = 1
        self.successful_placements = 0
        self._generate_new_harmony()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, self.game_over, self.steps >= self.MAX_STEPS, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0.0

        reward += self._handle_input(movement, space_held, shift_held)
        height_change_reward = self._update_physics_and_state()
        reward += height_change_reward
        self._update_particles()

        self.steps += 1
        terminated = self._is_win_condition() or self._is_loss_condition()
        truncated = self.steps >= self.MAX_STEPS
        self.game_over = terminated or truncated

        if terminated:
            if self._is_win_condition():
                reward += self.REWARD_WIN
            elif self._is_loss_condition():
                reward += self.REWARD_LOSS

        self.score += reward
        self.previous_space_held = space_held
        self.previous_shift_held = shift_held

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_input(self, movement, space_held, shift_held):
        event_reward = 0.0

        if movement == 3:  # Left
            self.player_vel.x = -self.PLAYER_SPEED
        elif movement == 4:  # Right
            self.player_vel.x = self.PLAYER_SPEED

        if movement == 1 and self.on_ground:  # Up (Jump)
            self.player_vel.y = -self.PLAYER_JUMP_FORCE

        if shift_held and not self.previous_shift_held:
            if len(self.selected_notes) < self.harmony_difficulty:
                next_note_index = len(self.selected_notes) % len(self.NOTE_PALETTE)
                self.selected_notes.append(next_note_index)
            elif self.selected_notes:  # Cycle the last note if list is not empty
                last_note = self.selected_notes[-1]
                self.selected_notes[-1] = (last_note + 1) % len(self.NOTE_PALETTE)

        if space_held and not self.previous_space_held:
            if self._check_harmony_match():
                event_reward += self.REWARD_PLATFORM_SUCCESS
                self._create_platform()
                self._create_particles(self.player_pos, (255, 255, 200), 30, 4)
            else:
                self._create_particles(self.player_pos, self.COLOR_FAILURE, 15, 2)
            self.selected_notes = []

        return event_reward

    def _update_physics_and_state(self):
        self.player_vel.x *= self.FRICTION
        if abs(self.player_vel.x) < 0.1:
            self.player_vel.x = 0

        self.player_vel.y += self.GRAVITY
        self.player_pos += self.player_vel

        self.on_ground = False
        player_rect = pygame.Rect(self.player_pos.x - self.PLAYER_SIZE / 2, self.player_pos.y - self.PLAYER_SIZE, self.PLAYER_SIZE, self.PLAYER_SIZE)
        for platform in self.platforms:
            if self.player_vel.y > 0 and player_rect.colliderect(platform["rect"]):
                if (self.player_pos.y - self.player_vel.y) <= platform["rect"].top:
                    self.player_pos.y = platform["rect"].top
                    self.player_vel.y = 0
                    self.on_ground = True
                    break

        self.player_pos.x = np.clip(self.player_pos.x, self.PLAYER_SIZE / 2, self.SCREEN_WIDTH - self.PLAYER_SIZE / 2)

        target_camera_y = self.player_pos.y - self.SCREEN_HEIGHT * 0.6
        self.camera_y = lerp(self.camera_y, target_camera_y, 0.1)
        self.camera_y = max(0, self.camera_y)

        height_change_reward = 0
        if self.player_pos.y < self.highest_y:
            pixels_climbed = self.highest_y - self.player_pos.y
            height_change_reward = pixels_climbed * self.REWARD_PER_PIXEL_CLIMBED
            self.highest_y = self.player_pos.y

        return height_change_reward

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['lifespan'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            p['size'] = max(0, p['size'] * 0.95)

    def _is_win_condition(self):
        # FIX: Original condition was `self.player_pos.y <= self.SUMMIT_HEIGHT - self.SCREEN_HEIGHT`
        # which evaluated to `350 <= 1200` (True) on reset, causing immediate termination.
        # The win condition is reaching the summit at y-coordinate 0 or less.
        return self.player_pos.y <= 0

    def _is_loss_condition(self):
        return self.player_pos.y > self.camera_y + self.SCREEN_HEIGHT + 50

    def _generate_new_harmony(self):
        self.target_harmony = self.np_random.integers(0, len(self.NOTE_PALETTE), size=self.harmony_difficulty).tolist()

    def _check_harmony_match(self):
        return self.selected_notes == self.target_harmony

    def _create_platform(self):
        platform_width = max(60, 150 - self.harmony_difficulty * 20)
        new_platform_rect = pygame.Rect(
            self.player_pos.x - platform_width / 2,
            self.player_pos.y - 5,
            platform_width,
            self.PLATFORM_HEIGHT
        )
        self.platforms.append({"rect": new_platform_rect, "notes": list(self.target_harmony)})

        if len(self.platforms) > self.MAX_PLATFORMS + 1:
            self.platforms.pop(1)

        self.successful_placements += 1
        if self.successful_placements % 2 == 0:
            self.harmony_difficulty = min(self.MAX_HARMONY_LENGTH, self.harmony_difficulty + 1)

        self._generate_new_harmony()

    def _get_observation(self):
        self._render_background()
        self._render_mountain()
        self._render_game_elements()
        self._render_particles()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for y in range(self.SCREEN_HEIGHT):
            interp = y / self.SCREEN_HEIGHT
            color = lerp(np.array(self.COLOR_BG_TOP), np.array(self.COLOR_BG_BOTTOM), interp)
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))

    def _render_mountain(self):
        for i in range(len(self.NOTE_PALETTE)):
            color = self.NOTE_PALETTE[i]
            x = (i + 1) * (self.SCREEN_WIDTH / (len(self.NOTE_PALETTE) + 1))
            offset = math.sin(self.steps * 0.01 + i) * 20
            w = 40 + offset
            h = self.SUMMIT_HEIGHT
            s = pygame.Surface((w, h), pygame.SRCALPHA)
            s.fill((color[0], color[1], color[2], 30))
            self.screen.blit(s, (x - w/2, self.SUMMIT_HEIGHT - h - self.camera_y))

    def _render_game_elements(self):
        for platform in self.platforms:
            p_rect = platform["rect"]
            draw_rect = p_rect.copy()
            draw_rect.y -= self.camera_y
            pygame.draw.rect(self.screen, (200, 200, 220), draw_rect, border_radius=3)
            pygame.draw.rect(self.screen, (150, 150, 170), draw_rect, width=2, border_radius=3)

            note_size = 8
            total_width = len(platform["notes"]) * (note_size + 2)
            start_x = draw_rect.centerx - total_width / 2
            for i, note_index in enumerate(platform["notes"]):
                color = self.NOTE_PALETTE[note_index]
                pygame.draw.rect(self.screen, color, (start_x + i * (note_size + 2), draw_rect.centery - note_size/2, note_size, note_size), border_radius=2)

        player_screen_pos = (int(self.player_pos.x), int(self.player_pos.y - self.camera_y))
        self._draw_glowing_circle(self.screen, self.COLOR_PLAYER_GLOW, player_screen_pos, self.PLAYER_SIZE, 15)
        pygame.gfxdraw.filled_circle(self.screen, player_screen_pos[0], player_screen_pos[1], int(self.PLAYER_SIZE * 0.7), self.COLOR_PLAYER)

    def _render_particles(self):
        for p in self.particles:
            pos = (int(p['pos'].x), int(p['pos'].y - self.camera_y))
            alpha = int(255 * (p['lifespan'] / p['max_lifespan']))
            color = (p['color'][0], p['color'][1], p['color'][2], alpha)

            particle_surf = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(particle_surf, color, (int(p['size']), int(p['size'])), int(p['size']))
            self.screen.blit(particle_surf, (pos[0] - p['size'], pos[1] - p['size']))

    def _render_ui(self):
        target_text = self.font_ui.render("TARGET", True, self.COLOR_UI_TEXT)
        self.screen.blit(target_text, (self.SCREEN_WIDTH / 2 - target_text.get_width() / 2, 10))

        note_size = 20
        spacing = 5
        total_width = len(self.target_harmony) * (note_size + spacing)
        start_x = self.SCREEN_WIDTH / 2 - total_width / 2
        for i, note_index in enumerate(self.target_harmony):
            color = self.NOTE_PALETTE[note_index]
            rect = pygame.Rect(start_x + i * (note_size + spacing), 40, note_size, note_size)
            pygame.draw.rect(self.screen, color, rect, border_radius=4)
            pygame.draw.rect(self.screen, (255, 255, 255), rect, width=2, border_radius=4)

        player_screen_y = self.player_pos.y - self.camera_y
        total_width = len(self.selected_notes) * (note_size + spacing)
        start_x = self.player_pos.x - total_width / 2
        for i, note_index in enumerate(self.selected_notes):
            color = self.NOTE_PALETTE[note_index]
            rect = pygame.Rect(start_x + i * (note_size + spacing), player_screen_y + 20, note_size, note_size)
            pygame.draw.rect(self.screen, color, rect, border_radius=4)

        height_climbed = max(0, self.SUMMIT_HEIGHT - self.player_pos.y - self.SCREEN_HEIGHT)
        height_text = self.font_ui.render(f"Height: {int(height_climbed)}m / {int(self.SUMMIT_HEIGHT - self.SCREEN_HEIGHT)}m", True, self.COLOR_UI_TEXT)
        self.screen.blit(height_text, (10, 10))

        score_text = self.font_ui.render(f"Score: {self.score:.1f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 30))

    def _create_particles(self, pos, color, count, speed):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            s = random.uniform(0.5, 1.0) * speed
            vel = pygame.Vector2(math.cos(angle) * s, math.sin(angle) * s)
            lifespan = random.randint(20, 40)
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'color': color,
                'lifespan': lifespan,
                'max_lifespan': lifespan,
                'size': random.uniform(3, 8)
            })

    def _draw_glowing_circle(self, surface, color, pos, radius, glow_strength):
        for i in range(glow_strength, 0, -1):
            alpha = int(255 * (1 - (i / glow_strength))**2 * 0.3)
            pygame.gfxdraw.filled_circle(
                surface, int(pos[0]), int(pos[1]),
                int(radius + i),
                (color[0], color[1], color[2], alpha)
            )

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "height": max(0, self.SUMMIT_HEIGHT - self.player_pos.y - self.SCREEN_HEIGHT),
            "harmony_difficulty": self.harmony_difficulty
        }

    def close(self):
        pygame.quit()


if __name__ == "__main__":
    # The __main__ block is for human play and is not part of the environment's API.
    # It requires a display, so we unset the dummy video driver.
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()

    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Harmony Ascent")
    clock = pygame.time.Clock()
    running = True

    while running:
        action = [0, 0, 0]  # [movement, space, shift]

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        elif keys[pygame.K_UP] or keys[pygame.K_w]:
            action[0] = 1

        if keys[pygame.K_SPACE]:
            action[1] = 1

        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Height: {info['height']:.0f}")
            obs, info = env.reset()

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(30)

    env.close()