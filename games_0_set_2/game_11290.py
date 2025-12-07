import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T16:54:32.754607
# Source Brief: brief_01290.md
# Brief Index: 1290
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

# Helper classes for game objects
class Pest:
    def __init__(self, col, row, color, pest_id, speed):
        self.col = col
        self.row = row
        self.color = color
        self.id = pest_id
        self.pixel_x = col * 64 + 32
        self.pixel_y = row * 40 - 40 # Start off-screen
        self.target_y = row * 40 + 20
        self.is_frozen = False
        self.bob_offset = random.uniform(0, 2 * math.pi)

    def update(self, speed):
        if not self.is_frozen:
            self.pixel_y += speed
            self.bob_offset += 0.1

    def draw(self, surface):
        y_pos = self.pixel_y + math.sin(self.bob_offset) * 2
        body_rect = pygame.Rect(self.pixel_x - 12, y_pos - 12, 24, 24)
        pygame.draw.rect(surface, self.color, body_rect, border_radius=4)
        pygame.draw.rect(surface, (0,0,0), body_rect, width=2, border_radius=4)
        # Eyes
        pygame.draw.circle(surface, (255,255,255), (int(self.pixel_x - 5), int(y_pos - 3)), 3)
        pygame.draw.circle(surface, (255,255,255), (int(self.pixel_x + 5), int(y_pos - 3)), 3)
        pygame.draw.circle(surface, (0,0,0), (int(self.pixel_x - 5), int(y_pos - 3)), 1)
        pygame.draw.circle(surface, (0,0,0), (int(self.pixel_x + 5), int(y_pos - 3)), 1)


class Particle:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.color = color
        self.vx = random.uniform(-2, 2)
        self.vy = random.uniform(-2, 2)
        self.life = random.randint(20, 40)
        self.size = random.randint(4, 8)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.life -= 1
        self.size = max(0, self.size - 0.2)

    def draw(self, surface):
        if self.life > 0:
            pygame.draw.rect(surface, self.color, (self.x, self.y, self.size, self.size))

class FloatingText:
    def __init__(self, x, y, text, font, color):
        self.x = x
        self.y = y
        self.text = text
        self.font = font
        self.color = color
        self.life = 60 # 2 seconds at 30fps

    def update(self):
        self.y -= 0.5
        self.life -= 1

    def draw(self, surface):
        if self.life > 0:
            alpha = max(0, min(255, self.life * 5))
            text_surf = self.font.render(self.text, True, self.color)
            text_surf.set_alpha(alpha)
            surface.blit(text_surf, (self.x, self.y))

class Trap:
    def __init__(self, col, row):
        self.col = col
        self.row = row
        self.pixel_x = col * 64 + 32
        self.pixel_y = row * 40 + 20
        self.radius = 48
        self.life = 150 # 5 seconds at 30fps

    def update(self):
        self.life -= 1

    def draw(self, surface):
        alpha = max(0, min(255, self.life * 2))
        color = (100, 200, 255, alpha)
        pygame.gfxdraw.filled_circle(surface, int(self.pixel_x), int(self.pixel_y), self.radius, color)
        pygame.gfxdraw.aacircle(surface, int(self.pixel_x), int(self.pixel_y), self.radius, color)


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Protect your garden by matching same-colored pests to clear them. "
        "Use special items like traps and sprays to handle overwhelming waves."
    )
    user_guide = (
        "Use arrow keys (↑↓←→) to move the cursor. Press 'space' to match pests or use a selected item. "
        "Press 'shift' to cycle through your items."
    )
    auto_advance = True

    # --- CONSTANTS ---
    WIDTH, HEIGHT = 640, 400
    GRID_COLS, GRID_ROWS = 10, 8
    CELL_WIDTH, CELL_HEIGHT = WIDTH // GRID_COLS, (HEIGHT - 80) // GRID_ROWS
    UI_HEIGHT = 80
    GAME_AREA_HEIGHT = HEIGHT - UI_HEIGHT
    MAX_STEPS = 2500
    MAX_WAVES = 20

    COLOR_BG = (40, 50, 30)
    COLOR_GRID = (60, 70, 50)
    COLOR_UI_BG = (30, 30, 40)
    COLOR_UI_TEXT = (255, 255, 220)
    COLOR_CURSOR = (180, 255, 255)

    PEST_COLORS = {
        "red": (255, 80, 80),
        "green": (80, 255, 80),
        "blue": (80, 120, 255),
        "yellow": (255, 255, 80),
        "purple": (200, 80, 255)
    }

    ITEM_TYPES = ["trap", "spray_red", "spray_green", "spray_blue"]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_float = pygame.font.SysFont("monospace", 18, bold=True)

        self.pest_id_counter = 0
        self.pests = []
        self.particles = []
        self.floating_texts = []
        self.active_traps = []

        self.cursor_pos = [0, 0]
        self.cursor_cooldown = 0
        self.space_was_held = False
        self.shift_was_held = False

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.wave_number = 1
        self.pest_speed = 1.0
        self.spawning_wave = False

        self.pests.clear()
        self.particles.clear()
        self.floating_texts.clear()
        self.active_traps.clear()

        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        self.cursor_cooldown = 0

        self.space_was_held = True # Prevent action on first frame
        self.shift_was_held = True

        self.inventory = {item: 0 for item in self.ITEM_TYPES}
        self.inventory["trap"] = 2
        self.unlocked_items = ["trap"]
        self.selected_item_idx = -1 # -1 means no item selected

        self._spawn_wave()

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0.0

        self._handle_input(action)

        # --- Update Game State ---
        self._update_traps()
        pest_penalty = self._update_pests()
        reward += pest_penalty

        self.particles = [p for p in self.particles if p.life > 0]
        for p in self.particles: p.update()

        self.floating_texts = [t for t in self.floating_texts if t.life > 0]
        for t in self.floating_texts: t.update()

        if hasattr(self, 'action_reward') and self.action_reward > 0:
            reward += self.action_reward
            self.action_reward = 0

        # --- Check State Transitions ---
        if not self.pests and not self.spawning_wave and not self.game_over:
            reward += 5.0 # Wave clear bonus
            self._add_floating_text(self.WIDTH // 2, self.HEIGHT // 2, "WAVE CLEAR! +5", (255, 255, 100))
            self.wave_number += 1

            if self.wave_number > self.MAX_WAVES:
                self.game_over = True
                reward += 100.0 # Win bonus
            else:
                if self.wave_number > 1 and self.wave_number % 2 == 1:
                    self.pest_speed = min(3.0, self.pest_speed + 0.1)
                if self.wave_number > 1 and self.wave_number % 4 == 1:
                    self._unlock_item()
                self._spawn_wave()

        if not self.game_over:
            for pest in self.pests:
                if pest.pixel_y > self.GAME_AREA_HEIGHT:
                    self.game_over = True
                    reward -= 100.0 # Lose penalty
                    break

        self.steps += 1
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        truncated = self.steps >= self.MAX_STEPS

        self.score += reward

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_input(self, action):
        self.action_reward = 0
        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1

        # --- Cursor Movement ---
        if self.cursor_cooldown <= 0:
            if movement == 1: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1) # Up
            elif movement == 2: self.cursor_pos[1] = min(self.GRID_ROWS - 1, self.cursor_pos[1] + 1) # Down
            elif movement == 3: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1) # Left
            elif movement == 4: self.cursor_pos[0] = min(self.GRID_COLS - 1, self.cursor_pos[0] + 1) # Right
            if movement != 0: self.cursor_cooldown = 4 # 4 frames cooldown
        else:
            self.cursor_cooldown -= 1

        # --- Action Buttons (Rising Edge Detection) ---
        space_activated = space_pressed and not self.space_was_held
        shift_activated = shift_pressed and not self.shift_was_held

        if shift_activated:
            # Cycle through available items
            if not self.unlocked_items:
                self.selected_item_idx = -1
            else:
                self.selected_item_idx = (self.selected_item_idx + 1) % (len(self.unlocked_items) + 1) -1

        if space_activated:
            if self.selected_item_idx != -1:
                self._use_item()
            else:
                self._match_pests()

        self.space_was_held = space_pressed
        self.shift_was_held = shift_pressed

    def _update_traps(self):
        self.active_traps = [t for t in self.active_traps if t.life > 0]
        for trap in self.active_traps:
            trap.update()
            for pest in self.pests:
                dist = math.hypot(pest.pixel_x - trap.pixel_x, pest.pixel_y - trap.pixel_y)
                if dist < trap.radius:
                    pest.is_frozen = True
                else: # Need to unfreeze if it moves out
                    pest.is_frozen = False

    def _update_pests(self):
        penalty = 0.0
        pests_to_remove = []
        for pest in self.pests:
            old_y = pest.pixel_y
            pest.update(self.pest_speed)
            # Check if pest crossed a row boundary
            if int(old_y / self.CELL_HEIGHT) != int(pest.pixel_y / self.CELL_HEIGHT):
                 penalty -= 0.1
            if pest.pixel_y > self.HEIGHT + 20:
                pests_to_remove.append(pest)

        self.pests = [p for p in self.pests if p not in pests_to_remove]

        if self.spawning_wave and all(p.pixel_y >= 0 for p in self.pests):
            self.spawning_wave = False
        return penalty

    def _spawn_wave(self):
        self.spawning_wave = True
        num_pests = min(20, 5 + self.wave_number * 2)

        # Ensure some matches are possible
        guaranteed_matches = min(num_pests // 4, 3)
        available_coords = [(c, r) for c in range(self.GRID_COLS) for r in range(self.GRID_ROWS)]
        random.shuffle(available_coords)

        pests_to_add = []

        for _ in range(guaranteed_matches):
            if len(available_coords) < 2: break
            c1, r1 = available_coords.pop()

            # Find a valid neighbor
            neighbors = [(c1+1, r1), (c1-1, r1), (c1, r1+1), (c1, r1-1)]
            valid_neighbors = [n for n in neighbors if n in available_coords]
            if not valid_neighbors: continue
            c2, r2 = random.choice(valid_neighbors)
            available_coords.remove((c2,r2))

            color_name = random.choice(list(self.PEST_COLORS.keys()))
            color_val = self.PEST_COLORS[color_name]

            pests_to_add.append(Pest(c1, r1, color_val, self.pest_id_counter, self.pest_speed)); self.pest_id_counter+=1
            pests_to_add.append(Pest(c2, r2, color_val, self.pest_id_counter, self.pest_speed)); self.pest_id_counter+=1

        # Fill remaining spots
        for _ in range(num_pests - len(pests_to_add)):
            if not available_coords: break
            c, r = available_coords.pop()
            color_name = random.choice(list(self.PEST_COLORS.keys()))
            color_val = self.PEST_COLORS[color_name]
            pests_to_add.append(Pest(c, r, color_val, self.pest_id_counter, self.pest_speed)); self.pest_id_counter+=1

        self.pests.extend(pests_to_add)

    def _unlock_item(self):
        potential_unlocks = [item for item in self.ITEM_TYPES if item not in self.unlocked_items]
        if potential_unlocks:
            new_item = random.choice(potential_unlocks)
            self.unlocked_items.append(new_item)
            self.inventory[new_item] = 1
            # sfx: unlock sound
            self._add_floating_text(self.WIDTH // 2, self.HEIGHT - self.UI_HEIGHT - 20, f"UNLOCKED: {new_item.upper()}!", (255, 215, 0))
        else: # Grant a random existing item if all are unlocked
            item_to_add = random.choice(self.unlocked_items)
            self.inventory[item_to_add] = self.inventory.get(item_to_add, 0) + 1

    def _use_item(self):
        item_name = self.unlocked_items[self.selected_item_idx]
        if self.inventory.get(item_name, 0) > 0:
            self.inventory[item_name] -= 1
            # sfx: item use sound

            pests_affected = 0
            if item_name == "trap":
                self.active_traps.append(Trap(self.cursor_pos[0], self.cursor_pos[1]))
                # Traps don't give immediate reward
            else: # Sprays
                color_to_spray = self.PEST_COLORS[item_name.split('_')[1]]
                pests_to_remove = [p for p in self.pests if p.color == color_to_spray]
                pests_affected = len(pests_to_remove)
                for p in pests_to_remove:
                    self._create_explosion(p.pixel_x, p.pixel_y, p.color)
                self.pests = [p for p in self.pests if p not in pests_to_remove]

            if pests_affected > 3:
                self.action_reward = 10.0
                self._add_floating_text(self.cursor_pos[0] * self.CELL_WIDTH, self.cursor_pos[1] * self.CELL_HEIGHT, f"GOOD! +10", (100, 255, 100))
            elif pests_affected > 0:
                self.action_reward = float(pests_affected)
                self._add_floating_text(self.cursor_pos[0] * self.CELL_WIDTH, self.cursor_pos[1] * self.CELL_HEIGHT, f"+{pests_affected}", (200, 200, 200))

            # De-select item after use
            self.selected_item_idx = -1

    def _match_pests(self):
        cursor_x = self.cursor_pos[0] * self.CELL_WIDTH + self.CELL_WIDTH // 2
        cursor_y = self.cursor_pos[1] * self.CELL_HEIGHT + self.CELL_HEIGHT // 2

        clicked_pest = None
        for pest in self.pests:
            dist = math.hypot(pest.pixel_x - cursor_x, pest.pixel_y - cursor_y)
            if dist < self.CELL_WIDTH // 2:
                clicked_pest = pest
                break

        if clicked_pest:
            to_process = deque([clicked_pest])
            matched_pests = {clicked_pest.id}

            while to_process:
                current_pest = to_process.popleft()
                for neighbor in self._get_neighbors(current_pest):
                    if neighbor.id not in matched_pests and neighbor.color == current_pest.color:
                        matched_pests.add(neighbor.id)
                        to_process.append(neighbor)

            if len(matched_pests) > 1:
                # sfx: match/explosion sound
                pests_to_remove = [p for p in self.pests if p.id in matched_pests]
                for p in pests_to_remove:
                    self._create_explosion(p.pixel_x, p.pixel_y, p.color)
                self.pests = [p for p in self.pests if p.id not in matched_pests]

                reward_val = len(matched_pests)
                self.action_reward = float(reward_val)
                self._add_floating_text(clicked_pest.pixel_x, clicked_pest.pixel_y, f"+{reward_val}", (255, 255, 255))

    def _get_neighbors(self, pest):
        neighbors = []
        for other_pest in self.pests:
            if pest.id == other_pest.id: continue
            dist = math.hypot(pest.pixel_x - other_pest.pixel_x, pest.pixel_y - other_pest.pixel_y)
            if dist < max(self.CELL_WIDTH, self.CELL_HEIGHT) * 0.8: # Adjacency threshold
                neighbors.append(other_pest)
        return neighbors

    def _create_explosion(self, x, y, color):
        for _ in range(20):
            self.particles.append(Particle(x, y, color))

    def _add_floating_text(self, x, y, text, color):
        self.floating_texts.append(FloatingText(x, y, text, self.font_float, color))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for i in range(1, self.GRID_COLS):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i * self.CELL_WIDTH, 0), (i * self.CELL_WIDTH, self.GAME_AREA_HEIGHT))
        for i in range(1, self.GRID_ROWS):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i * self.CELL_HEIGHT), (self.WIDTH, i * self.CELL_HEIGHT))

        for trap in self.active_traps: trap.draw(self.screen)
        for pest in self.pests: pest.draw(self.screen)
        for particle in self.particles: particle.draw(self.screen)

        # Draw cursor
        cursor_rect = pygame.Rect(self.cursor_pos[0] * self.CELL_WIDTH, self.cursor_pos[1] * self.CELL_HEIGHT, self.CELL_WIDTH, self.CELL_HEIGHT)

        # Pulsing glow effect for cursor
        glow_alpha = 100 + math.sin(pygame.time.get_ticks() * 0.005) * 50
        glow_surf = pygame.Surface((self.CELL_WIDTH, self.CELL_HEIGHT), pygame.SRCALPHA)
        glow_color = (*self.COLOR_CURSOR, glow_alpha)
        pygame.draw.rect(glow_surf, glow_color, glow_surf.get_rect(), border_radius=8)
        self.screen.blit(glow_surf, cursor_rect.topleft)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 3, border_radius=8)

        for text in self.floating_texts: text.draw(self.screen)

    def _render_ui(self):
        ui_rect = pygame.Rect(0, self.GAME_AREA_HEIGHT, self.WIDTH, self.UI_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, ui_rect)
        pygame.draw.line(self.screen, self.COLOR_UI_TEXT, (0, self.GAME_AREA_HEIGHT), (self.WIDTH, self.GAME_AREA_HEIGHT), 2)

        score_text = self.font_ui.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, self.GAME_AREA_HEIGHT + 10))

        wave_text = self.font_ui.render(f"WAVE: {self.wave_number}/{self.MAX_WAVES}", True, self.COLOR_UI_TEXT)
        self.screen.blit(wave_text, (self.WIDTH - wave_text.get_width() - 10, self.GAME_AREA_HEIGHT + 10))

        # Render inventory
        start_x = 10
        y_pos = self.GAME_AREA_HEIGHT + 40
        for i, item_name in enumerate(self.unlocked_items):
            count = self.inventory.get(item_name, 0)
            item_text = f"[{item_name.split('_')[0].upper()}: {count}]"

            is_selected = (i == self.selected_item_idx)
            color = self.COLOR_CURSOR if is_selected else self.COLOR_UI_TEXT

            text_surf = self.font_ui.render(item_text, True, color)
            self.screen.blit(text_surf, (start_x, y_pos))
            start_x += text_surf.get_width() + 15

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0,0))

            status = "YOU WIN!" if self.wave_number > self.MAX_WAVES else "GAME OVER"
            status_font = pygame.font.SysFont("monospace", 60, bold=True)
            status_text = status_font.render(status, True, (255, 50, 50))
            text_rect = status_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(status_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave_number,
            "pests_on_screen": len(self.pests),
            "inventory": self.inventory,
        }

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()

    # Un-dummy the video driver for manual play
    os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "macOS"
    pygame.quit()
    pygame.init()

    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Pixel Garden Protector")
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

        # The environment returns an array, so we need to convert it back for display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Total Reward: {total_reward:.2f}, Steps: {info['steps']}")
            # Wait a bit before resetting
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        clock.tick(30) # Run at 30 FPS

    env.close()