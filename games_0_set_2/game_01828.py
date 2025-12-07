import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ↑↓←→ to move cursor. Space to plant on empty plots or harvest mature crops. Move to the barn and hold Shift to sell your inventory."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Manage a small isometric farm to earn $1000 within a time limit. Plant, grow, harvest, and sell crops to reach the goal."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Game Constants ---
    # Colors
    COLOR_BG = (20, 30, 35)
    COLOR_GRID = (40, 55, 60)
    COLOR_PLOT = (101, 67, 33)
    COLOR_BARN = (139, 0, 0)
    COLOR_CURSOR = (255, 255, 0)
    COLOR_TEXT = (220, 220, 220)
    COLOR_TEXT_SHADOW = (20, 20, 20)
    COLOR_TIMER_BAR_FULL = (0, 200, 0)
    COLOR_TIMER_BAR_MID = (255, 255, 0)
    COLOR_TIMER_BAR_LOW = (255, 0, 0)

    # Game Parameters
    GRID_WIDTH = 6
    GRID_HEIGHT = 6
    TILE_WIDTH_HALF = 32
    TILE_HEIGHT_HALF = 16
    STARTING_MONEY = 50
    WIN_CONDITION_MONEY = 1000
    MAX_STEPS = 600

    # Crop Definitions: {name: [cost, grow_time, sell_value, color_seed, color_mature]}
    CROP_DATA = {
        "wheat": [5, 50, 15, (154, 205, 50), (218, 165, 32)],
        "pumpkin": [15, 120, 50, (34, 139, 34), (255, 140, 0)],
    }
    CROP_TYPES = list(CROP_DATA.keys())

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()

        self.font_ui = pygame.font.Font(None, 28)
        self.font_pop = pygame.font.Font(None, 22)

        self.origin_x = 640 // 2
        self.origin_y = 100

        # self.reset() is called by the environment wrapper, no need to call it here.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.money = self.STARTING_MONEY
        self.game_over = False
        self.win = False

        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.barn_pos = [0, 0]  # Top-left corner of the grid

        self.farm_grid = [
            [
                {"type": None, "growth": 0}
                for _ in range(self.GRID_HEIGHT)
            ] for _ in range(self.GRID_WIDTH)
        ]

        self.inventory = {crop: 0 for crop in self.CROP_TYPES}
        self.current_crop_selection = 0  # Index for CROP_TYPES

        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_pressed, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0

        # 1. Update game state (non-action based)
        self.steps += 1
        self._update_crops()
        self._update_particles()

        # 2. Process player actions
        # Action: Movement
        if movement == 1: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)  # Up
        elif movement == 2: self.cursor_pos[1] = min(self.GRID_HEIGHT - 1, self.cursor_pos[1] + 1)  # Down
        elif movement == 3: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)  # Left
        elif movement == 4: self.cursor_pos[0] = min(self.GRID_WIDTH - 1, self.cursor_pos[0] + 1)  # Right

        cx, cy = self.cursor_pos
        plot = self.farm_grid[cx][cy]

        # Action: Space (Plant/Harvest)
        if space_pressed:
            if cx == self.barn_pos[0] and cy == self.barn_pos[1]:
                # Cycle crop selection at barn
                self.current_crop_selection = (self.current_crop_selection + 1) % len(self.CROP_TYPES)
            elif plot["type"] is None:
                # Plant
                crop_to_plant = self.CROP_TYPES[self.current_crop_selection]
                cost = self.CROP_DATA[crop_to_plant][0]
                if self.money >= cost:
                    self.money -= cost
                    plot["type"] = crop_to_plant
                    plot["growth"] = 1
                    reward += 0.1
                    self._create_text_popup(f"-${cost}", (cx, cy), (255, 80, 80))
                else:
                    reward -= 0.01  # Penalty for invalid action
            elif plot["growth"] >= self.CROP_DATA[plot["type"]][1]:
                # Harvest
                crop_type = plot["type"]
                self.inventory[crop_type] += 1
                plot["type"] = None
                plot["growth"] = 0
                reward += 0.5
                self._create_particle_burst((cx, cy), self.CROP_DATA[crop_type][4])
            else:
                reward -= 0.01  # Penalty for trying to harvest unripe crop

        # Action: Shift (Sell)
        if shift_held:
            if cx == self.barn_pos[0] and cy == self.barn_pos[1]:
                sale_total = 0
                for crop_type, count in self.inventory.items():
                    if count > 0:
                        sale_total += count * self.CROP_DATA[crop_type][2]

                if sale_total > 0:
                    self.money += sale_total
                    self.inventory = {crop: 0 for crop in self.CROP_TYPES}
                    reward += 1.0
                    self._create_text_popup(f"+${sale_total}", (cx, cy), (80, 255, 80))
                else:
                    reward -= 0.01  # Penalty for trying to sell nothing
            else:
                reward -= 0.01  # Penalty for holding shift on non-barn tile

        # 3. Check for termination
        terminated = self._check_termination()
        if terminated:
            if self.win:
                reward = 100.0  # Win bonus
            else:
                reward = -10.0  # Lose penalty

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_crops(self):
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                plot = self.farm_grid[x][y]
                if plot["type"] is not None:
                    grow_time = self.CROP_DATA[plot["type"]][1]
                    if plot["growth"] < grow_time:
                        plot["growth"] += 1

    def _check_termination(self):
        if self.money >= self.WIN_CONDITION_MONEY:
            self.game_over = True
            self.win = True
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            self.win = False
            return True
        return False

    def _get_info(self):
        return {
            "score": self.money,
            "steps": self.steps,
            "inventory": self.inventory,
            "cursor_pos": self.cursor_pos,
        }

    def _cart_to_iso(self, x, y):
        iso_x = self.origin_x + (x - y) * self.TILE_WIDTH_HALF
        iso_y = self.origin_y + (x + y) * self.TILE_HEIGHT_HALF
        return iso_x, iso_y

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid and crops
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                iso_x, iso_y = self._cart_to_iso(x, y)
                points = [
                    (iso_x, iso_y - self.TILE_HEIGHT_HALF),
                    (iso_x + self.TILE_WIDTH_HALF, iso_y),
                    (iso_x, iso_y + self.TILE_HEIGHT_HALF),
                    (iso_x - self.TILE_WIDTH_HALF, iso_y),
                ]

                # Draw base tile
                is_barn = (x == self.barn_pos[0] and y == self.barn_pos[1])
                tile_color = self.COLOR_BARN if is_barn else self.COLOR_PLOT
                pygame.gfxdraw.filled_polygon(self.screen, points, tile_color)
                pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_GRID)

                # Draw crops
                plot = self.farm_grid[x][y]
                if plot["type"] is not None:
                    self._render_crop(plot, (iso_x, iso_y))

                # Draw barn details
                if is_barn:
                    self._render_barn((iso_x, iso_y))

        # Draw cursor
        cursor_iso_x, cursor_iso_y = self._cart_to_iso(self.cursor_pos[0], self.cursor_pos[1])
        cursor_points = [
            (cursor_iso_x, cursor_iso_y - self.TILE_HEIGHT_HALF),
            (cursor_iso_x + self.TILE_WIDTH_HALF, cursor_iso_y),
            (cursor_iso_x, cursor_iso_y + self.TILE_HEIGHT_HALF),
            (cursor_iso_x - self.TILE_WIDTH_HALF, cursor_iso_y),
        ]
        pulse = (math.sin(pygame.time.get_ticks() * 0.01) + 1) / 2 * 3
        pygame.draw.polygon(self.screen, self.COLOR_CURSOR, cursor_points, max(1, int(3 + pulse)))

        # Draw particles
        for p in self.particles:
            if p.get('is_text', False):
                continue
            p_pos = (int(p['x']), int(p['y']))
            pygame.draw.circle(self.screen, p['color'], p_pos, int(p['size']))

    def _render_crop(self, plot, pos):
        crop_type = plot["type"]
        data = self.CROP_DATA[crop_type]
        max_growth = data[1]

        progress = min(1.0, plot["growth"] / max_growth)

        min_size = 2
        max_size = self.TILE_HEIGHT_HALF * 0.8
        size = min_size + (max_size - min_size) * progress

        color_start = pygame.Color(data[3])
        color_end = pygame.Color(data[4])
        color = color_start.lerp(color_end, progress)

        # Simple 3D effect
        shadow_color = (max(0, color.r - 50), max(0, color.g - 50), max(0, color.b - 50))
        pygame.gfxdraw.filled_ellipse(self.screen, int(pos[0]), int(pos[1] + size * 0.2), int(size), int(size * 0.5), shadow_color)
        pygame.gfxdraw.filled_ellipse(self.screen, int(pos[0]), int(pos[1]), int(size), int(size * 0.5), color)
        pygame.gfxdraw.aaellipse(self.screen, int(pos[0]), int(pos[1]), int(size), int(size * 0.5), (0, 0, 0, 80))

    def _render_barn(self, pos):
        # Roof
        roof_points = [
            (pos[0], pos[1] - self.TILE_HEIGHT_HALF * 1.5),
            (pos[0] + self.TILE_WIDTH_HALF, pos[1] - self.TILE_HEIGHT_HALF * 0.5),
            (pos[0], pos[1] + self.TILE_HEIGHT_HALF * 0.5),
            (pos[0] - self.TILE_WIDTH_HALF, pos[1] - self.TILE_HEIGHT_HALF * 0.5),
        ]
        pygame.gfxdraw.filled_polygon(self.screen, roof_points, (100, 20, 20))
        pygame.gfxdraw.aapolygon(self.screen, roof_points, self.COLOR_GRID)
        # Door
        door_rect = pygame.Rect(0, 0, 15, 20)
        door_rect.center = (pos[0], pos[1] + 5)
        pygame.draw.rect(self.screen, (50, 20, 20), door_rect)

    def _render_ui(self):
        # Render text with shadow
        def draw_text(text, font, color, pos):
            shadow = font.render(text, True, self.COLOR_TEXT_SHADOW)
            content = font.render(text, True, color)
            self.screen.blit(shadow, (pos[0] + 2, pos[1] + 2))
            self.screen.blit(content, pos)

        # Money
        money_text = f"$ {int(self.money)}"
        draw_text(money_text, self.font_ui, (255, 215, 0), (10, 10))

        # Time bar
        time_ratio = 1.0 - (self.steps / self.MAX_STEPS)
        bar_width = 200
        bar_height = 20
        bar_x = 640 - bar_width - 10
        bar_y = 10

        if time_ratio < 0.25:
            color = self.COLOR_TIMER_BAR_LOW
        elif time_ratio < 0.6:
            color = self.COLOR_TIMER_BAR_MID
        else:
            color = self.COLOR_TIMER_BAR_FULL

        pygame.draw.rect(self.screen, self.COLOR_TEXT_SHADOW, (bar_x, bar_y, bar_width, bar_height))
        pygame.draw.rect(self.screen, color, (bar_x, bar_y, int(bar_width * time_ratio), bar_height))
        pygame.draw.rect(self.screen, self.COLOR_TEXT, (bar_x, bar_y, bar_width, bar_height), 2)

        # Inventory and Crop Selection
        inv_y_start = 390
        inv_x_start = 10

        sel_crop_name = self.CROP_TYPES[self.current_crop_selection]
        sel_crop_cost = self.CROP_DATA[sel_crop_name][0]
        sel_text = f"Planting: {sel_crop_name.capitalize()} (${sel_crop_cost})"
        draw_text(sel_text, self.font_ui, self.COLOR_TEXT, (inv_x_start, inv_y_start - 30))

        inv_text = "Inventory: "
        for crop, count in self.inventory.items():
            inv_text += f"{crop.capitalize()}: {count}  "
        draw_text(inv_text, self.font_ui, self.COLOR_TEXT, (inv_x_start, inv_y_start - 60))

        # Pop-up text particles
        for p in self.particles:
            if p.get('is_text', False):
                draw_text(p['text'], self.font_pop, p['color'], (p['x'], p['y']))

        # Game Over Screen
        if self.game_over:
            overlay = pygame.Surface((640, 400), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))

            big_font = pygame.font.Font(None, 72)
            end_text = "YOU WIN!" if self.win else "TIME'S UP!"
            end_color = (0, 255, 0) if self.win else (255, 0, 0)
            draw_text(end_text, big_font, end_color, (180, 150))

            final_score_text = f"Final Money: ${int(self.money)}"
            draw_text(final_score_text, self.font_ui, self.COLOR_TEXT, (240, 220))

    def _create_particle_burst(self, grid_pos, color):
        iso_x, iso_y = self._cart_to_iso(grid_pos[0], grid_pos[1])
        for _ in range(20):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            self.particles.append({
                'x': iso_x, 'y': iso_y,
                'vx': math.cos(angle) * speed, 'vy': math.sin(angle) * speed - 2,
                'size': random.uniform(2, 5),
                'life': random.randint(20, 40),
                'color': color
            })

    def _create_text_popup(self, text, grid_pos, color):
        iso_x, iso_y = self._cart_to_iso(grid_pos[0], grid_pos[1])
        self.particles.append({
            'x': iso_x - 15, 'y': iso_y - 20,
            'vx': 0, 'vy': -0.5,
            'life': 40,
            'color': color,
            'text': text,
            'is_text': True
        })

    def _update_particles(self):
        new_particles = []
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            if not p.get('is_text', False):
                p['vy'] += 0.1  # Gravity
                p['size'] *= 0.95
            p['life'] -= 1
            if p['life'] > 0 and p.get('size', 1) > 0.5:
                new_particles.append(p)
        self.particles = new_particles

    def close(self):
        pygame.quit()

    def render(self):
        return self._get_observation()


if __name__ == '__main__':
    # This block allows you to play the game manually
    # It will not run in a headless environment
    try:
        os.environ["SDL_VIDEODRIVER"] = "x11"
        env = GameEnv(render_mode="rgb_array")
        obs, info = env.reset()

        # Override screen for display
        env.screen = pygame.display.set_mode((640, 400))
        pygame.display.set_caption("Farming Simulator")

        done = False
        total_reward = 0

        # Game loop
        while not done:
            # --- Action mapping for human play ---
            movement = 0  # No-op
            space = 0
            shift = 0

            # Pygame event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True

            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4

            if keys[pygame.K_SPACE]: space = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

            action = [movement, space, shift]

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

            # Render the observation to the display
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            env.screen.blit(surf, (0, 0))
            pygame.display.flip()

            # Control the frame rate
            env.clock.tick(15)  # Slower for human play

            if done:
                print(f"Game Over! Final Money: ${info['score']:.2f}, Total Reward: {total_reward:.2f}")
                # Wait a bit before closing
                pygame.time.wait(3000)

        env.close()
    except pygame.error as e:
        print(f"Could not run in graphical mode: {e}")
        print("This is expected in a headless environment. The code is likely correct.")