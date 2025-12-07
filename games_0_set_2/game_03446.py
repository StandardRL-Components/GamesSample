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

    # Short, user-facing control string
    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press Space to place a crystal. "
        "Hold Shift to cycle through crystal types."
    )

    # Short, user-facing description of the game
    game_description = (
        "An isometric puzzle game. Strategically place crystals to trigger chain reactions and collect all the gems before you run out of moves."
    )

    # Frames auto-advance for smooth animations
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 16, 10
    TILE_WIDTH_HALF, TILE_HEIGHT_HALF = 28, 14
    ORIGIN_X = SCREEN_WIDTH // 2
    ORIGIN_Y = 80

    # Colors
    COLOR_BG = (26, 28, 44)
    COLOR_GRID = (40, 42, 60)
    COLOR_GRID_HI = (60, 62, 90)
    COLOR_GEM = (255, 220, 0)
    COLOR_GEM_SPARKLE = (255, 255, 180)
    COLOR_CURSOR = (0, 255, 255)
    CRYSTAL_COLORS = [
        (255, 0, 128),  # Magenta (Horizontal)
        (0, 255, 128),  # Green (Vertical)
        (255, 128, 0),  # Orange (Area of Effect)
    ]
    CRYSTAL_NAMES = ["Horizontal", "Vertical", "Area"]

    # Game parameters
    MAX_MOVES = 15
    GEMS_TO_WIN = 20
    TOTAL_GEMS = 30
    ANIMATION_DURATION = 30  # frames

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        self.render_mode = render_mode

        # Pygame setup
        pygame.init()
        pygame.font.init()
        # FIX: Initialize a display mode. This is required for surface operations
        # like .convert_alpha() to work, even in headless "dummy" mode.
        self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.Font(None, 28)
        self.font_large = pygame.font.Font(None, 72)
        self.font_small = pygame.font.Font(None, 20)

        # State variables are initialized in reset()
        self.reset()

        # Validate implementation
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)

        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]

        self.gems = set()
        while len(self.gems) < self.TOTAL_GEMS:
            gx, gy = self.rng.integers(0, self.GRID_WIDTH), self.rng.integers(0, self.GRID_HEIGHT)
            self.gems.add((gx, gy))

        self.crystals = []
        self.particles = []

        self.score = 0
        self.moves_remaining = self.MAX_MOVES
        self.game_over = False
        self.win = False

        self.game_phase = "player_turn"  # or "animation"
        self.animation_timer = 0
        self.last_reaction_data = None

        self.current_crystal_type = 0
        self.last_space_held = False
        self.last_shift_held = False

        self.steps = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        self.steps += 1
        reward = 0

        # Unpack factorized action
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        if self.game_phase == "player_turn" and not self.game_over:
            # --- Handle Input ---
            # Movement
            if movement == 1: self.cursor_pos[1] -= 1  # Up
            elif movement == 2: self.cursor_pos[1] += 1  # Down
            elif movement == 3: self.cursor_pos[0] -= 1  # Left
            elif movement == 4: self.cursor_pos[0] += 1  # Right
            self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_WIDTH - 1)
            self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_HEIGHT - 1)

            # Cycle crystal type
            if shift_held and not self.last_shift_held:
                self.current_crystal_type = (self.current_crystal_type + 1) % len(self.CRYSTAL_COLORS)
                # Sound: UI cycle

            # Place crystal
            if space_held and not self.last_space_held:
                crystal_pos = tuple(self.cursor_pos)
                is_occupied = any(c['pos'] == crystal_pos for c in self.crystals) or crystal_pos in self.gems
                if not is_occupied and self.moves_remaining > 0:
                    self.moves_remaining -= 1
                    # Sound: Crystal place

                    new_crystal = {
                        "pos": crystal_pos,
                        "type": self.current_crystal_type,
                        "color": self.CRYSTAL_COLORS[self.current_crystal_type]
                    }
                    self.crystals.append(new_crystal)

                    # Trigger chain reaction
                    reward += self._trigger_reaction(new_crystal)

                    self.game_phase = "animation"
                    self.animation_timer = self.ANIMATION_DURATION

        elif self.game_phase == "animation":
            self.animation_timer -= 1
            if self.animation_timer <= 0:
                self.game_phase = "player_turn"
                self.last_reaction_data = None  # Clear old reaction visuals

        # Update input trackers
        self.last_space_held = space_held
        self.last_shift_held = shift_held

        # Update particles
        self._update_particles()

        # Check for termination
        terminated = False
        if self.game_phase == "player_turn" and not self.game_over:
            if self.score >= self.GEMS_TO_WIN:
                self.win = True
                self.game_over = True
                terminated = True
                reward += 50  # Win bonus
                # Sound: Win
            elif self.moves_remaining <= 0:
                self.win = False
                self.game_over = True
                terminated = True
                # Sound: Lose

        if self.auto_advance:
            self.clock.tick(30)

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _trigger_reaction(self, crystal):
        gems_hit = set()
        affected_tiles = set()

        x, y = crystal['pos']

        if crystal['type'] == 0:  # Horizontal
            for i in range(self.GRID_WIDTH):
                affected_tiles.add((i, y))
        elif crystal['type'] == 1:  # Vertical
            for i in range(self.GRID_HEIGHT):
                affected_tiles.add((x, i))
        elif crystal['type'] == 2:  # Area of Effect
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT:
                        affected_tiles.add((nx, ny))

        for tile in affected_tiles:
            if tile in self.gems:
                gems_hit.add(tile)

        # Store data for rendering the animation
        self.last_reaction_data = {
            "origin": crystal['pos'],
            "type": crystal['type'],
            "color": crystal['color'],
            "affected_tiles": affected_tiles,
            "gems_hit": gems_hit,
        }

        # Update state and calculate reward
        gems_collected_count = 0
        for gem_pos in gems_hit:
            self.gems.remove(gem_pos)
            self.score += 1
            gems_collected_count += 1
            # Sound: Gem collect
            # Create sparkle effect
            screen_pos = self._iso_to_cart(gem_pos[0], gem_pos[1])
            for _ in range(20):
                self._create_particle(
                    pos=list(screen_pos),
                    vel=[(self.rng.random() - 0.5) * 4, (self.rng.random() - 0.5) * 4],
                    life=self.rng.integers(15, 25),
                    color=self.COLOR_GEM_SPARKLE,
                    radius_start=self.rng.integers(2, 5),
                    radius_end=0
                )

        return gems_collected_count

    def _create_particle(self, pos, vel, life, color, radius_start, radius_end):
        self.particles.append({
            "pos": pos, "vel": vel, "life": life, "max_life": life,
            "color": color, "radius_start": radius_start, "radius_end": radius_end
        })

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['life'] -= 1
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "moves_remaining": self.moves_remaining}

    def _iso_to_cart(self, grid_x, grid_y):
        screen_x = self.ORIGIN_X + (grid_x - grid_y) * self.TILE_WIDTH_HALF
        screen_y = self.ORIGIN_Y + (grid_x + grid_y) * self.TILE_HEIGHT_HALF
        return int(screen_x), int(screen_y)

    def _draw_iso_poly(self, surface, color, grid_x, grid_y, filled=True):
        px, py = self._iso_to_cart(grid_x, grid_y)
        points = [
            (px, py - self.TILE_HEIGHT_HALF),
            (px + self.TILE_WIDTH_HALF, py),
            (px, py + self.TILE_HEIGHT_HALF),
            (px - self.TILE_WIDTH_HALF, py),
        ]
        if filled:
            pygame.gfxdraw.filled_polygon(surface, points, color)
        pygame.gfxdraw.aapolygon(surface, points, color)

    def _render_game(self):
        # Draw grid
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                self._draw_iso_poly(self.screen, self.COLOR_GRID, x, y, filled=False)

        # Draw gems
        for gx, gy in self.gems:
            px, py = self._iso_to_cart(gx, gy)
            pulse = (math.sin(pygame.time.get_ticks() * 0.005 + gx + gy) + 1) / 2
            size = int(3 + pulse * 3)
            pygame.gfxdraw.filled_circle(self.screen, px, py - self.TILE_HEIGHT_HALF // 2, size, self.COLOR_GEM)
            pygame.gfxdraw.aacircle(self.screen, px, py - self.TILE_HEIGHT_HALF // 2, size, self.COLOR_GEM)

        # Draw crystals
        for c in self.crystals:
            self._draw_iso_poly(self.screen, c['color'], c['pos'][0], c['pos'][1], filled=True)

        # Draw chain reaction animation
        if self.game_phase == "animation" and self.last_reaction_data:
            progress = 1.0 - (self.animation_timer / self.ANIMATION_DURATION)
            data = self.last_reaction_data
            r, g, b = data['color']

            for tile_pos in data['affected_tiles']:
                alpha = int(max(0, 255 * (math.sin(progress * math.pi))))
                temp_surf = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
                self._draw_iso_poly(temp_surf, (r, g, b, alpha), tile_pos[0], tile_pos[1])
                self.screen.blit(temp_surf, (0, 0))

        # Draw cursor
        if self.game_phase == "player_turn" and not self.game_over:
            cursor_color = self.CRYSTAL_COLORS[self.current_crystal_type]
            pulse = (math.sin(pygame.time.get_ticks() * 0.01) + 1) / 2
            final_color = (
                int(cursor_color[0] * (0.5 + pulse * 0.5)),
                int(cursor_color[1] * (0.5 + pulse * 0.5)),
                int(cursor_color[2] * (0.5 + pulse * 0.5))
            )
            self._draw_iso_poly(self.screen, final_color, self.cursor_pos[0], self.cursor_pos[1], filled=False)

        # Draw particles
        for p in self.particles:
            progress = p['life'] / p['max_life']
            radius = int(p['radius_end'] + (p['radius_start'] - p['radius_end']) * progress)
            if radius > 0:
                # Create a temporary surface for alpha drawing
                temp_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, (*p['color'], int(255 * progress)), (radius, radius), radius)
                self.screen.blit(temp_surf, (int(p['pos'][0]) - radius, int(p['pos'][1]) - radius))


    def _render_ui(self):
        # Helper to draw text with shadow
        def draw_text(text, font, color, pos, center=False):
            text_surf = font.render(text, True, color)
            shadow_surf = font.render(text, True, (0, 0, 0))
            text_rect = text_surf.get_rect()
            if center:
                text_rect.center = pos
            else:
                text_rect.topleft = pos

            self.screen.blit(shadow_surf, (text_rect.x + 2, text_rect.y + 2))
            self.screen.blit(text_surf, text_rect)

        # Draw score and moves
        draw_text(f"Gems: {self.score} / {self.GEMS_TO_WIN}", self.font_main, (255, 255, 255), (10, 10))
        draw_text(f"Moves: {self.moves_remaining}", self.font_main, (255, 255, 255),
                  (self.SCREEN_WIDTH - 130, 10))

        # Draw current crystal type UI
        if not self.game_over:
            crystal_name = self.CRYSTAL_NAMES[self.current_crystal_type]
            crystal_color = self.CRYSTAL_COLORS[self.current_crystal_type]
            draw_text("Crystal:", self.font_small, (200, 200, 200), (10, self.SCREEN_HEIGHT - 45))
            draw_text(crystal_name, self.font_small, crystal_color, (70, self.SCREEN_HEIGHT - 45))
            draw_text("(Shift to cycle)", self.font_small, (150, 150, 150), (10, self.SCREEN_HEIGHT - 25))

        # Draw game over message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            if self.win:
                draw_text("YOU WIN!", self.font_large, self.COLOR_GEM,
                          (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2), center=True)
            else:
                draw_text("GAME OVER", self.font_large, (200, 50, 50),
                          (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2), center=True)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]

        # Test observation space
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8

        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)

        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc is False
        assert isinstance(info, dict)

        print("✓ Implementation validated successfully")


# Example of how to run the environment
if __name__ == '__main__':
    # Set this to "human" to play the game with keyboard controls
    # Set to "rgb_array" to run in headless mode (for testing)
    render_mode = "human"

    if render_mode == "human":
        # NOTE: The human mode in this example script may not work correctly because
        # SDL_VIDEODRIVER is set to "dummy" at the top of the file, preventing
        # a real window from being created. The fix is focused on the GameEnv class.
        
        # Remap keys for human play
        key_map = {
            pygame.K_UP: 1,
            pygame.K_DOWN: 2,
            pygame.K_LEFT: 3,
            pygame.K_RIGHT: 4,
        }

        env = GameEnv(render_mode="rgb_array")
        obs, info = env.reset()

        # Create a display window for human mode
        display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        pygame.display.set_caption("Crystal Caverns")

        terminated = False
        while not terminated:
            # Construct action from keyboard state
            keys = pygame.key.get_pressed()

            movement_action = 0  # No-op
            for key, action_val in key_map.items():
                if keys[key]:
                    movement_action = action_val
                    break

            space_action = 1 if keys[pygame.K_SPACE] else 0
            shift_action = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

            action = [movement_action, space_action, shift_action]

            obs, reward, terminated, truncated, info = env.step(action)

            # Render the observation to the display window
            frame = np.transpose(obs, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame)
            display_screen.blit(surf, (0, 0))
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True

        print(f"Game Over. Final Score: {info['score']}")
        # Keep window open for a few seconds to show final state
        pygame.time.wait(3000)
        env.close()

    else:  # rgb_array mode for testing
        env = GameEnv(render_mode="rgb_array")
        obs, info = env.reset()
        terminated = False
        total_reward = 0
        step_count = 0

        while not terminated and step_count < 1000:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
            if reward > 0:
                print(f"Step {step_count}: Reward={reward}, Total Reward={total_reward}, Info={info}")

        print(f"Finished random play after {step_count} steps.")
        print(f"Final Score: {info['score']}, Total Reward: {total_reward}")
        env.close()