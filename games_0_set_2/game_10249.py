from gymnasium.spaces import MultiDiscrete
import os
import pygame


import gymnasium as gym
from gymnasium.spaces import Box, MultiDiscrete
import numpy as np
import os
import pygame
import pygame.gfxdraw

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T10:04:57.491157
# Source Brief: brief_00249.md
# Brief Index: 249
# """import gymnasium as gym
from gymnasium.spaces import Box, MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment for a dream-like puzzle game.

    The player must clone and place "dream creatures" onto a grid to match a
    target pattern. This must be done while avoiding "nightmare guards" that
    patrol the area. The game combines strategy, puzzle-solving, and stealth.

    **Visuals:**
    - Ethereal, high-contrast aesthetic.
    - Player is a bright, glowing entity.
    - Dream creatures are pastel-colored shapes.
    - Nightmare guards are dark, menacing shapes with a visible detection aura.
    - The target pattern is shown as faint "ghosts" on the grid.
    - UI is clean and displays score, selected creature, and controls.

    **Gameplay:**
    - The player moves freely around the grid.
    - Pressing 'Space' cycles through the available dream creature types.
    - Pressing 'Shift' places the currently selected creature at the player's location.
    - Placing the correct creature in a target spot scores points.
    - Getting caught by a guard ends the game.
    - Completing the entire pattern wins the game.
    - Difficulty increases as the score rises, with more and faster guards.
    """
    metadata = {"render_modes": ["rgb_array"]}
    game_description = (
        "Place dream creatures to complete a target pattern while avoiding patrolling nightmare guards in this ethereal puzzle game."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move. Press space to cycle through creature types and shift to place a creature."
    )
    auto_advance = True


    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.CELL_SIZE = 20
        self.GRID_W = self.WIDTH // self.CELL_SIZE
        self.GRID_H = self.HEIGHT // self.CELL_SIZE
        self.MAX_STEPS = 1000
        self.NUM_CREATURE_TYPES = 3
        self.TARGET_PLACEMENTS = 15

        # --- Colors ---
        self.COLOR_BG = (15, 10, 25)
        self.COLOR_GRID = (30, 20, 50)
        self.COLOR_PLAYER = (0, 255, 255)
        self.COLOR_PLAYER_GLOW = (0, 128, 128)
        self.COLOR_GUARD = (120, 40, 100)
        self.COLOR_GUARD_AURA = self.COLOR_GUARD + (50,)
        self.CREATURE_COLORS = [
            (255, 100, 100),  # Pastel Red
            (100, 255, 100),  # Pastel Green
            (100, 100, 255),  # Pastel Blue
        ]
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_SUCCESS = (100, 255, 100, 150)
        self.COLOR_FAIL = (255, 100, 100, 150)

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
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)

        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = [0.0, 0.0]
        self.player_speed = 4.0
        self.guards = []
        self.guard_base_speed = 1.0
        self.guard_detection_radius = 40
        self.board_grid = None
        self.target_grid = None
        self.creature_counts = {}
        self.selected_creature_type = 1
        self.last_space_held = False
        self.last_shift_held = False
        self.particles = []
        self.feedback_effects = []

        # The original code called reset() here, but it's better to let the
        # user/runner call it explicitly. The state is initialized above.

    def _generate_target_and_inventory(self):
        self.target_grid = np.zeros((self.GRID_W, self.GRID_H), dtype=int)
        self.creature_counts = {i + 1: 0 for i in range(self.NUM_CREATURE_TYPES)}
        
        placed_targets = 0
        while placed_targets < self.TARGET_PLACEMENTS:
            x, y = self.np_random.integers(0, self.GRID_W), self.np_random.integers(0, self.GRID_H)
            if self.target_grid[x, y] == 0:
                creature_type = self.np_random.integers(1, self.NUM_CREATURE_TYPES + 1)
                self.target_grid[x, y] = creature_type
                self.creature_counts[creature_type] += 1
                placed_targets += 1

    def _add_guard(self):
        path_type = self.np_random.choice(['rect', 'horizontal', 'vertical'])
        if path_type == 'rect':
            x1 = self.np_random.integers(1, self.GRID_W // 2 - 2)
            x2 = self.np_random.integers(self.GRID_W // 2 + 2, self.GRID_W - 2)
            y1 = self.np_random.integers(1, self.GRID_H // 2 - 2)
            y2 = self.np_random.integers(self.GRID_H // 2 + 2, self.GRID_H - 2)
            path = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        elif path_type == 'horizontal':
            y = self.np_random.integers(1, self.GRID_H - 1)
            x1 = self.np_random.integers(1, self.GRID_W // 2 - 5)
            x2 = self.np_random.integers(self.GRID_W // 2 + 5, self.GRID_W - 2)
            path = [(x1, y), (x2, y)]
        else:  # vertical
            x = self.np_random.integers(1, self.GRID_W - 1)
            y1 = self.np_random.integers(1, self.GRID_H // 2 - 5)
            y2 = self.np_random.integers(self.GRID_H // 2 + 5, self.GRID_H - 2)
            path = [(x, y1), (x, y2)]

        start_pos = [p * self.CELL_SIZE + self.CELL_SIZE / 2 for p in path[0]]
        self.guards.append({
            'pos': start_pos, 'path': path, 'path_idx': 0, 'speed': self.guard_base_speed
        })

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = [self.WIDTH / 2, self.HEIGHT / 2]
        self.board_grid = np.zeros((self.GRID_W, self.GRID_H), dtype=int)
        self._generate_target_and_inventory()
        self.guards = []
        self.guard_base_speed = 1.0
        self._add_guard()
        self.selected_creature_type = 1
        self.last_space_held = False
        self.last_shift_held = False
        self.particles = []
        self.feedback_effects = []
        return self._get_observation(), self._get_info()

    def _handle_input(self, movement, space_pressed, shift_pressed):
        # Movement
        if movement == 1: self.player_pos[1] -= self.player_speed
        if movement == 2: self.player_pos[1] += self.player_speed
        if movement == 3: self.player_pos[0] -= self.player_speed
        if movement == 4: self.player_pos[0] += self.player_speed
        self.player_pos[0] = np.clip(self.player_pos[0], 0, self.WIDTH - 1)
        self.player_pos[1] = np.clip(self.player_pos[1], 0, self.HEIGHT - 1)

        # Cycle selected creature
        if space_pressed:
            # Sfx: UI_switch
            self.selected_creature_type = (self.selected_creature_type % self.NUM_CREATURE_TYPES) + 1

        # Place creature
        return self._place_creature() if shift_pressed else 0.0

    def _place_creature(self):
        grid_x = int(self.player_pos[0] / self.CELL_SIZE)
        grid_y = int(self.player_pos[1] / self.CELL_SIZE)

        if self.board_grid[grid_x, grid_y] == 0 and self.creature_counts[self.selected_creature_type] > 0:
            self.board_grid[grid_x, grid_y] = self.selected_creature_type
            self.creature_counts[self.selected_creature_type] -= 1
            
            if self.target_grid[grid_x, grid_y] == self.selected_creature_type:
                # Sfx: placement_success
                self.score += 1
                self.feedback_effects.append({'pos': (grid_x, grid_y), 'color': self.COLOR_SUCCESS, 'timer': 15})
                self._update_difficulty()
                return 1.0
            else:
                # Sfx: placement_fail
                self.feedback_effects.append({'pos': (grid_x, grid_y), 'color': self.COLOR_FAIL, 'timer': 15})
                return 0.0
        return 0.0

    def _update_difficulty(self):
        num_guards_expected = 1 + (self.score // 25)
        if len(self.guards) < num_guards_expected:
             self._add_guard()
        
        new_speed = 1.0 + (self.score // 50) * 0.1
        if new_speed > self.guard_base_speed:
            self.guard_base_speed = new_speed
            for guard in self.guards:
                guard['speed'] = self.guard_base_speed

    def _update_guards(self):
        for guard in self.guards:
            target_grid_pos = guard['path'][guard['path_idx']]
            target_pos = np.array([p * self.CELL_SIZE + self.CELL_SIZE / 2 for p in target_grid_pos])
            direction = target_pos - np.array(guard['pos'])
            dist = np.linalg.norm(direction)
            
            if dist < guard['speed']:
                guard['pos'] = list(target_pos)
                guard['path_idx'] = (guard['path_idx'] + 1) % len(guard['path'])
            else:
                guard['pos'] += (direction / dist) * guard['speed']

            if np.linalg.norm(np.array(self.player_pos) - np.array(guard['pos'])) < self.guard_detection_radius:
                # Sfx: caught_by_guard
                for _ in range(50): self.particles.append(self._create_particle(self.player_pos, self.COLOR_GUARD))
                return True
        return False

    def _update_effects(self):
        self.particles = [p for p in self.particles if p['lifespan'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
        
        self.feedback_effects = [f for f in self.feedback_effects if f['timer'] > 0]
        for f in self.feedback_effects: f['timer'] -= 1

    def _create_particle(self, pos, color):
        return {
            'pos': np.array(pos, dtype=float),
            'vel': self.np_random.uniform(-2, 2, size=2),
            'lifespan': self.np_random.integers(20, 40),
            'color': color,
            'size': self.np_random.uniform(2, 5)
        }

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = -0.01  # Small time penalty

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.last_space_held
        shift_pressed = shift_held and not self.last_shift_held

        reward += self._handle_input(movement, space_pressed, shift_pressed)
        
        caught = self._update_guards()
        self._update_effects()
        self.steps += 1
        
        correct_placements = np.sum((self.board_grid > 0) & (self.board_grid == self.target_grid))
        won = correct_placements == self.TARGET_PLACEMENTS

        terminated = False
        truncated = False
        if caught:
            reward -= 5.0
            terminated = True
        elif won:
            # Sfx: game_win
            reward += 50.0
            terminated = True
        elif self.steps >= self.MAX_STEPS:
            truncated = True # Use truncated for time limits

        self.game_over = terminated or truncated
        self.last_space_held, self.last_shift_held = space_held, shift_held
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    # --- Rendering Methods ---
    def _render_game(self):
        self._render_grid()
        self._render_target_ghosts()
        self._render_feedback_effects()
        self._render_placed_creatures()
        self._render_guards()
        self._render_player()
        self._render_particles()

    def _render_grid(self):
        for x in range(0, self.WIDTH, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

    def _render_target_ghosts(self):
        for x in range(self.GRID_W):
            for y in range(self.GRID_H):
                c_type = self.target_grid[x, y]
                if c_type > 0 and self.board_grid[x, y] != c_type:
                    color = self.CREATURE_COLORS[c_type - 1]
                    center = (x * self.CELL_SIZE + self.CELL_SIZE // 2, y * self.CELL_SIZE + self.CELL_SIZE // 2)
                    s = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
                    self._draw_creature_shape(s, c_type, (self.CELL_SIZE//2, self.CELL_SIZE//2), color, self.CELL_SIZE*0.3)
                    s.set_alpha(40)
                    self.screen.blit(s, (x * self.CELL_SIZE, y * self.CELL_SIZE))

    def _render_placed_creatures(self):
        for x in range(self.GRID_W):
            for y in range(self.GRID_H):
                c_type = self.board_grid[x, y]
                if c_type > 0:
                    color = self.CREATURE_COLORS[c_type - 1]
                    center = (x * self.CELL_SIZE + self.CELL_SIZE // 2, y * self.CELL_SIZE + self.CELL_SIZE // 2)
                    self._draw_creature_shape(self.screen, c_type, center, color, self.CELL_SIZE * 0.4)

    def _draw_creature_shape(self, surface, c_type, center, color, size):
        x, y, size = int(center[0]), int(center[1]), int(size)
        if c_type == 1: # Circle
            pygame.gfxdraw.aacircle(surface, x, y, size, color)
            pygame.gfxdraw.filled_circle(surface, x, y, size, color)
        elif c_type == 2: # Square
            pygame.draw.rect(surface, color, (x - size, y - size, size * 2, size * 2))
        elif c_type == 3: # Triangle
            points = [(x, y - size), (x - size, y + size // 2), (x + size, y + size // 2)]
            pygame.gfxdraw.aapolygon(surface, points, color)
            pygame.gfxdraw.filled_polygon(surface, points, color)

    def _render_guards(self):
        for guard in self.guards:
            pos = (int(guard['pos'][0]), int(guard['pos'][1]))
            radius = int(self.guard_detection_radius)
            s = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(s, radius, radius, radius, self.COLOR_GUARD_AURA)
            self.screen.blit(s, (pos[0]-radius, pos[1]-radius))
            size = 10
            points = [(pos[0], pos[1] - size), (pos[0] + size // 2, pos[1]), (pos[0], pos[1] + size), (pos[0] - size // 2, pos[1])]
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_GUARD)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_GUARD)

    def _render_player(self):
        if self.game_over: return
        pos = (int(self.player_pos[0]), int(self.player_pos[1]))
        glow_radius = self.CELL_SIZE // 2 + 3
        s = pygame.Surface((glow_radius*2, glow_radius*2), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(s, glow_radius, glow_radius, glow_radius, self.COLOR_PLAYER_GLOW + (80,))
        self.screen.blit(s, (pos[0]-glow_radius, pos[1]-glow_radius))
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.CELL_SIZE // 2, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.CELL_SIZE // 2, self.COLOR_PLAYER)

    def _render_particles(self):
        for p in self.particles: pygame.draw.circle(self.screen, p['color'], p['pos'].astype(int), int(p['size']))

    def _render_feedback_effects(self):
        for f in self.feedback_effects:
            alpha = int(f['timer'] / 15 * f['color'][3])
            color = f['color'][:3] + (alpha,)
            radius = int(self.CELL_SIZE * 0.7 * (1.0 - f['timer']/15.0))
            s = pygame.Surface((self.CELL_SIZE*2, self.CELL_SIZE*2), pygame.SRCALPHA)
            pygame.gfxdraw.aacircle(s, self.CELL_SIZE, self.CELL_SIZE, radius, color)
            self.screen.blit(s, (f['pos'][0] * self.CELL_SIZE - self.CELL_SIZE//2, f['pos'][1] * self.CELL_SIZE- self.CELL_SIZE//2))

    def _render_ui(self):
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        ui_x, ui_y = self.WIDTH - 150, 20
        pygame.draw.rect(self.screen, self.COLOR_GRID, (ui_x - 10, ui_y - 10, 140, 80), border_radius=5)
        c_type, c_color = self.selected_creature_type, self.CREATURE_COLORS[self.selected_creature_type - 1]
        self._draw_creature_shape(self.screen, c_type, (ui_x + 20, ui_y + 30), c_color, 15)
        count_text = self.font_large.render(f"x {self.creature_counts[c_type]}", True, self.COLOR_TEXT)
        self.screen.blit(count_text, (ui_x + 50, ui_y + 15))
        instr_text = self.font_small.render("[SPACE] Cycle [SHIFT] Place", True, self.COLOR_TEXT)
        self.screen.blit(instr_text, (ui_x - 5, ui_y + 55))

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0,0,0,180))
            self.screen.blit(overlay, (0,0))
            won = np.sum((self.board_grid > 0) & (self.board_grid == self.target_grid)) == self.TARGET_PLACEMENTS
            status_text = "LANDSCAPE COMPLETE" if won else "CAUGHT!"
            status_color = self.COLOR_SUCCESS if won else self.COLOR_FAIL
            end_text = self.font_large.render(status_text, True, status_color)
            self.screen.blit(end_text, end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2 - 20)))
            score_text = self.font_small.render(f"Final Score: {self.score}", True, self.COLOR_TEXT)
            self.screen.blit(score_text, score_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2 + 20)))

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It is not part of the required environment implementation
    
    # Re-enable video driver for human play
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # --- Manual Control Setup ---
    # Pygame window for human play
    render_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Dream Weaver")
    clock = pygame.time.Clock()

    action = [0, 0, 0]  # [movement, space, shift]

    print("\n--- Controls ---")
    print("Arrows: Move")
    print("Space: Cycle Creature")
    print("Shift: Place Creature")
    print("Q: Quit")
    print("R: Reset")
    print("----------------\n")

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    done = True
                if event.key == pygame.K_r:
                    obs, info = env.reset()

        # Get key presses for action
        keys = pygame.key.get_pressed()
        
        # Movement
        action[0] = 0 # No-op
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        # Space and Shift
        action[1] = 1 if keys[pygame.K_SPACE] else 0
        action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"Game Over! Score: {info['score']}, Steps: {info['steps']}")
            # Render final state
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            render_screen.blit(surf, (0, 0))
            pygame.display.flip()
            # Wait for a moment before resetting
            pygame.time.wait(2000)
            obs, info = env.reset()

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(30) # Run at 30 FPS

    env.close()