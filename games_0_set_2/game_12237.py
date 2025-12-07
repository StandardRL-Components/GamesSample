import gymnasium as gym
import os
import pygame
import numpy as np
from gymnasium.spaces import MultiDiscrete, Box
import pygame.gfxdraw
import math
import random
import string
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A Gymnasium environment for a neon-themed word-spelling defense game.
    The player must spell words to destroy approaching monsters before they
    reach and damage the city.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = "Defend your neon city from approaching monsters by spelling the words that appear above them."
    user_guide = "Use ←→ arrow keys to select letters, press space to type, and press shift to clear your current word."
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    MAX_STEPS = 5000
    CITY_HEALTH_MAX = 100
    TARGET_FPS = 30

    # --- Colors (Neon Aesthetic) ---
    COLOR_BG = (10, 5, 25)
    COLOR_GRID = (30, 20, 60)
    COLOR_SKYLINE = (20, 10, 40)
    COLOR_SKYLINE_NEON = (100, 80, 220)

    COLOR_PLAYER = (0, 255, 255)  # Cyan
    COLOR_PLAYER_GLOW = (100, 255, 255)

    COLOR_TEXT = (255, 255, 255)
    COLOR_SCORE = (255, 255, 0)  # Yellow
    COLOR_HEALTH_FG = (0, 255, 128)  # Bright Green
    COLOR_HEALTH_BG = (255, 0, 100)  # Hot Pink

    COLOR_CORRECT = (50, 255, 50)
    COLOR_INCORRECT = (255, 50, 50)

    # --- Word List ---
    WORD_LIST = {
        3: ["SKY", "RUN", "CAT", "DOG", "SUN", "FLY", "BUG", "BOT"],
        4: ["CODE", "GAME", "WORD", "NEON", "CITY", "GRID", "BEAM", "LAZR"],
        5: ["PULSE", "LASER", "VIRUS", "AGENT", "PROXY", "GLOWS", "BYTES"],
        6: ["ATTACK", "DEFEND", "SYSTEM", "PLAYER", "ACTION", "REWARD"],
        7: ["MONSTER", "CORRECT", "GYMNASIUM", "VECTOR", "PYTHON"],
        8: ["TERMINAL", "SEQUENCE", "FEEDBACK", "RENDERED", "ACCURACY"],
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.ui_font = pygame.font.SysFont("monospace", 18, bold=True)
        self.word_font = pygame.font.SysFont("monospace", 22, bold=True)
        self.monster_font = pygame.font.SysFont("monospace", 16, bold=True)

        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.city_health = self.CITY_HEALTH_MAX
        self.monsters = []
        self.particles = []
        self.effects = []
        self.screen_shake = 0

        self.player_spelling = ""
        self.available_tiles = []
        self.selected_tile_index = 0
        self.target_monster = None

        self.monster_base_speed = 0.5
        self.word_difficulty_level = 3

        self.last_space_held = 0
        self.last_shift_held = 0
        self.clear_charges = 3

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.city_health = self.CITY_HEALTH_MAX
        self.monsters = []
        self.particles = []
        self.effects = []
        self.screen_shake = 0

        self.player_spelling = ""
        self.available_tiles = []
        self.selected_tile_index = 0
        self.target_monster = None

        self.monster_base_speed = 0.5
        self.word_difficulty_level = 3

        self.last_space_held = 0
        self.last_shift_held = 0
        self.clear_charges = 3

        self._spawn_monster()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0.0

        # --- Handle Input ---
        movement, space_held, shift_held = action
        space_pressed = space_held and not self.last_space_held
        shift_pressed = shift_held and not self.last_shift_held
        self.last_space_held = space_held
        self.last_shift_held = shift_held

        reward += self._handle_input(movement, space_pressed, shift_pressed)

        # --- Update Game State ---
        self._update_difficulty()
        reward += self._update_monsters()
        self._update_particles_and_effects()

        # --- Check Termination ---
        truncated = self.steps >= self.MAX_STEPS
        terminated = self.city_health <= 0
        if terminated or truncated:
            self.game_over = True
            if self.city_health > 0 and truncated:
                reward += 50.0  # Survival bonus
                self.effects.append({"type": "text", "text": "SURVIVAL BONUS +50", "pos": (self.WIDTH // 2, self.HEIGHT // 2), "life": 60, "color": self.COLOR_SCORE})

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_input(self, movement, space_pressed, shift_pressed):
        reward = 0.0
        # --- Tile Navigation ---
        if movement == 3:  # Left
            self.selected_tile_index = (self.selected_tile_index - 1 + len(self.available_tiles)) % len(self.available_tiles) if self.available_tiles else 0
        elif movement == 4:  # Right
            self.selected_tile_index = (self.selected_tile_index + 1) % len(self.available_tiles) if self.available_tiles else 0

        # --- Select Letter (Space Press) ---
        if space_pressed and self.target_monster and self.available_tiles:
            selected_char = self.available_tiles[self.selected_tile_index]
            target_word = self.target_monster["word"]

            if len(self.player_spelling) < len(target_word) and selected_char == target_word[len(self.player_spelling)]:
                # Correct letter
                self.player_spelling += selected_char
                reward += 1.0
                # Sound: Correct beep
                pos = (self.WIDTH // 2, self.HEIGHT - 50)
                self.effects.append({"type": "flash", "pos": pos, "radius": 40, "life": 10, "max_life": 10, "color": self.COLOR_CORRECT})

                if self.player_spelling == target_word:
                    # Word complete
                    reward += 5.0
                    self.score += len(target_word) * 10
                    self._create_explosion(self.target_monster["pos"], self.target_monster["color"], 100)
                    self.monsters.remove(self.target_monster)
                    self.target_monster = None
                    self.player_spelling = ""
                    self._spawn_monster()
                    # Sound: Explosion
            else:
                # Incorrect letter
                reward -= 0.5
                self.screen_shake = 15
                self.player_spelling = ""  # Reset attempt on wrong letter
                # Sound: Error buzz
                self.effects.append({"type": "text", "text": "FAIL", "pos": (self.WIDTH // 2, self.HEIGHT // 2), "life": 30, "color": self.COLOR_INCORRECT})

        # --- Use Clear Charge (Shift Press) ---
        if shift_pressed and self.clear_charges > 0 and self.player_spelling:
            self.clear_charges -= 1
            self.player_spelling = ""
            # Sound: Power-up sound
            pos = (self.WIDTH // 2, self.HEIGHT - 50)
            self.effects.append({"type": "flash", "pos": pos, "radius": 50, "life": 20, "max_life": 20, "color": self.COLOR_PLAYER})

        return reward

    def _update_difficulty(self):
        # Increase word length every 100 steps
        if self.steps > 0 and self.steps % 100 == 0:
            self.word_difficulty_level = min(self.word_difficulty_level + 1, max(self.WORD_LIST.keys()))
        # Increase monster speed every 200 steps
        if self.steps > 0 and self.steps % 200 == 0:
            self.monster_base_speed = min(self.monster_base_speed + 0.1, 2.5)

    def _update_monsters(self):
        reward = 0.0
        monsters_to_remove = []
        for monster in self.monsters:
            monster["pos"].y += monster["speed"]
            if monster["pos"].y > self.HEIGHT - 70:  # Reached city defense line
                monsters_to_remove.append(monster)
                self.city_health = max(0, self.city_health - 25)
                reward -= 2.0
                self.screen_shake = 20
                # Sound: City damage / alarm
                self.effects.append({"type": "flash", "pos": (self.WIDTH // 2, self.HEIGHT - 35), "radius": 100, "life": 20, "max_life": 20, "color": self.COLOR_INCORRECT})

        for monster in monsters_to_remove:
            self.monsters.remove(monster)
            if self.target_monster == monster:
                self.target_monster = None
                self.player_spelling = ""

        if not self.monsters and not self.game_over:
            self._spawn_monster()

        return reward

    def _update_particles_and_effects(self):
        # Update particles
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)

        # Update general effects
        for e in self.effects[:]:
            e["life"] -= 1
            if e["life"] <= 0:
                self.effects.remove(e)

        if self.screen_shake > 0:
            self.screen_shake -= 1

    def _spawn_monster(self):
        difficulty = min(self.word_difficulty_level, max(self.WORD_LIST.keys()))
        word = self.np_random.choice(self.WORD_LIST[difficulty])

        pos = pygame.Vector2(self.np_random.uniform(100, self.WIDTH - 100), -20)
        speed = self.monster_base_speed + self.np_random.uniform(-0.1, 0.1)
        color = random.choice([(255, 0, 128), (255, 128, 0), (128, 0, 255)])  # Hot Pink, Orange, Purple

        monster = {
            "word": word,
            "pos": pos,
            "speed": speed,
            "color": color,
            "radius": 15 + len(word)
        }
        self.monsters.append(monster)

        # Set new target and generate tiles
        self.target_monster = monster
        self.player_spelling = ""
        self.selected_tile_index = 0

        # Generate tiles: word letters + distractors
        num_distractors = max(2, 8 - len(word))
        distractors = self.np_random.choice(list(string.ascii_uppercase), num_distractors, replace=False)

        tiles = list(word) + list(distractors)
        self.np_random.shuffle(tiles)
        self.available_tiles = tiles

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        render_offset = pygame.Vector2(0, 0)
        if self.screen_shake > 0:
            render_offset.x = self.np_random.uniform(-self.screen_shake, self.screen_shake) * 0.5
            render_offset.y = self.np_random.uniform(-self.screen_shake, self.screen_shake) * 0.5

        self._render_background(render_offset)
        self._render_particles(render_offset)
        self._render_monsters(render_offset)
        self._render_platform_and_tiles(render_offset)
        self._render_effects(render_offset)
        self._render_ui()  # UI is not affected by screen shake

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "city_health": self.city_health,
            "clear_charges": self.clear_charges,
        }

    # --- Rendering Methods ---

    def _render_background(self, offset):
        # Grid
        for i in range(0, self.WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i + int(offset.x), 0), (i + int(offset.x), self.HEIGHT))
        for i in range(0, self.HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i + int(offset.y)), (self.WIDTH, i + int(offset.y)))
        # Skyline
        for i in range(15):
            x = i * 45 + self.np_random.uniform(-10, 10)
            h = 50 + i % 5 * 20 + self.np_random.uniform(5, 30)
            w = 30 + self.np_random.uniform(-5, 5)
            rect = pygame.Rect(x + offset.x, self.HEIGHT - h + offset.y, w, h)
            pygame.draw.rect(self.screen, self.COLOR_SKYLINE, rect)
            pygame.draw.rect(self.screen, self.COLOR_SKYLINE_NEON, rect, 1)

    def _render_monsters(self, offset):
        for monster in self.monsters:
            pos = monster["pos"] + offset
            radius = monster["radius"]
            color = monster["color"]

            # Pulsating glow
            pulse = (math.sin(self.steps * 0.2) + 1) / 2
            glow_radius = radius * (1.2 + pulse * 0.4)
            self._draw_glow_circle(self.screen, color, pos, glow_radius, 4)

            # Core
            pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), int(radius), color)
            pygame.gfxdraw.aacircle(self.screen, int(pos.x), int(pos.y), int(radius), (255, 255, 255))

            # Word above monster
            word_surf = self.monster_font.render(monster["word"], True, self.COLOR_TEXT)
            word_rect = word_surf.get_rect(center=(pos.x, pos.y - radius - 15))
            self.screen.blit(word_surf, word_rect)

    def _render_platform_and_tiles(self, offset):
        # Platform base
        platform_y = self.HEIGHT - 35
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, (0 + offset.x, platform_y + offset.y, self.WIDTH, 35))
        pygame.draw.line(self.screen, self.COLOR_PLAYER_GLOW, (0 + offset.x, platform_y + offset.y), (self.WIDTH + offset.x, platform_y + offset.y), 3)

        # Current spelling
        spelling_surf = self.word_font.render(self.player_spelling, True, self.COLOR_BG)
        spelling_rect = spelling_surf.get_rect(center=(self.WIDTH // 2 + offset.x, self.HEIGHT - 55 + offset.y))
        self.screen.blit(spelling_surf, spelling_rect)

        # Available tiles
        if self.available_tiles:
            tile_width = 40
            total_width = len(self.available_tiles) * tile_width
            start_x = (self.WIDTH - total_width) // 2

            for i, char in enumerate(self.available_tiles):
                is_selected = (i == self.selected_tile_index)
                x = start_x + i * tile_width + tile_width // 2
                y = self.HEIGHT - 18

                color = self.COLOR_PLAYER_GLOW if is_selected else self.COLOR_PLAYER
                char_surf = self.ui_font.render(char, True, color)
                char_rect = char_surf.get_rect(center=(x + offset.x, y + offset.y))
                self.screen.blit(char_surf, char_rect)

                if is_selected:
                    pygame.draw.rect(self.screen, color, (char_rect.left - 4, char_rect.top - 2, char_rect.width + 8, char_rect.height + 4), 1)

    def _render_particles(self, offset):
        for p in self.particles:
            pos = p["pos"] + offset
            pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), int(p["radius"]), p["color"])

    def _render_effects(self, offset):
        for e in self.effects:
            if e["type"] == "flash":
                alpha = int(255 * (e["life"] / e["max_life"]))
                self._draw_glow_circle(self.screen, e["color"], e["pos"] + offset, e["radius"], 4, alpha)
            elif e["type"] == "text":
                alpha = int(255 * (e["life"] / 30)) if e["life"] < 30 else 255
                font_surf = self.word_font.render(e["text"], True, e["color"])
                font_surf.set_alpha(alpha)
                font_rect = font_surf.get_rect(center=e["pos"] + offset)
                self.screen.blit(font_surf, font_rect)

    def _render_ui(self):
        # Score
        score_surf = self.ui_font.render(f"SCORE: {self.score}", True, self.COLOR_SCORE)
        self.screen.blit(score_surf, (10, 10))

        # Steps / Time
        time_surf = self.ui_font.render(f"TIME: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        time_rect = time_surf.get_rect(centerx=self.WIDTH // 2)
        self.screen.blit(time_surf, (time_rect.x, 10))

        # Clear Charges
        charges_surf = self.ui_font.render(f"CLEAR: {self.clear_charges}", True, self.COLOR_PLAYER)
        charges_rect = charges_surf.get_rect(right=self.WIDTH - 10, top=30)
        self.screen.blit(charges_surf, charges_rect)

        # Health Bar
        health_ratio = self.city_health / self.CITY_HEALTH_MAX
        bar_width = 200
        bar_height = 15

        bg_rect = pygame.Rect(self.WIDTH - bar_width - 10, 10, bar_width, bar_height)
        fg_width = int(bar_width * health_ratio)
        fg_rect = pygame.Rect(self.WIDTH - bar_width - 10, 10, fg_width, bar_height)

        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BG, bg_rect)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_FG, fg_rect)
        pygame.draw.rect(self.screen, self.COLOR_TEXT, bg_rect, 1)

    # --- Helper Methods ---

    def _create_explosion(self, pos, color, num_particles):
        # Sound: Explosion
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 6)
            vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            self.particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "life": self.np_random.integers(20, 40),
                "radius": self.np_random.uniform(1, 4),
                "color": color
            })
        self.effects.append({"type": "flash", "pos": pos, "radius": 100, "life": 20, "max_life": 20, "color": (255, 255, 255)})
        self.screen_shake = 25

    def _draw_glow_circle(self, surface, color, center, radius, layers, alpha=255):
        temp_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        for i in range(layers, 0, -1):
            layer_alpha = int(alpha * (1 - (i / (layers + 1))) ** 2 * 0.5)
            layer_color = (*color, layer_alpha)
            pygame.gfxdraw.filled_circle(temp_surf, int(radius), int(radius), int(radius * i / layers), layer_color)
        surface.blit(temp_surf, (int(center.x - radius), int(center.y - radius)))

    def close(self):
        pygame.quit()

    def render(self):
        # This method is not used in the standard gym loop but is good practice to have.
        return self._get_observation()


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # Make sure to unset the dummy video driver if you want to see the game
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()

    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Word Defender")
    clock = pygame.time.Clock()

    terminated = False
    truncated = False
    total_reward = 0

    while not terminated and not truncated:
        movement, space, shift = 0, 0, 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4

        if keys[pygame.K_SPACE]:
            space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift = 1

        action = [movement, space, shift]

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Display the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(GameEnv.TARGET_FPS)

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}, Steps: {info['steps']}")
            # Wait a bit before closing
            pygame.time.wait(3000)

    env.close()