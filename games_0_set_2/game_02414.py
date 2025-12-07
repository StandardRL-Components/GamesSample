
# Generated: 2025-08-27T20:17:25.230184
# Source Brief: brief_02414.md
# Brief Index: 2414

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ↑↓←→ to move. Press space at the glowing ritual site to win after collecting all 5 items."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate a haunted graveyard, collecting ritual items while evading ghosts to complete a ritual."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    GRID_WIDTH = 32
    GRID_HEIGHT = 20
    CELL_SIZE = 20
    SCREEN_WIDTH = GRID_WIDTH * CELL_SIZE
    SCREEN_HEIGHT = GRID_HEIGHT * CELL_SIZE
    MAX_STEPS = 1000
    NUM_ITEMS = 5
    NUM_GHOSTS = 2

    # --- Colors ---
    COLOR_BG = (15, 10, 25)
    COLOR_PLAYER = (255, 255, 255)
    COLOR_GHOST_1 = (255, 50, 50)
    COLOR_GHOST_2 = (220, 20, 20)
    COLOR_WALL = (60, 70, 80)
    COLOR_RITUAL = (150, 0, 255)
    ITEM_COLORS = [
        (0, 255, 0),    # Green
        (0, 150, 255),  # Blue
        (255, 255, 0),  # Yellow
        (255, 120, 0),  # Orange
        (0, 255, 255),  # Cyan
    ]
    COLOR_UI_TEXT = (240, 240, 240)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)

        # Initialize state variables
        self.player_pos = None
        self.ghosts = []
        self.items = []
        self.items_collected = None
        self.walls = set()
        self.ritual_pos = None
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.np_random = None

        self._generate_level()
        self.validate_implementation()


    def _generate_level(self):
        """Creates the static elements of the level: walls and ritual site."""
        self.walls = set()
        # Borders
        for x in range(self.GRID_WIDTH):
            self.walls.add((x, 0))
            self.walls.add((x, self.GRID_HEIGHT - 1))
        for y in range(self.GRID_HEIGHT):
            self.walls.add((0, y))
            self.walls.add((self.GRID_WIDTH - 1, y))

        # Gravestones/Obstacles
        gravestones = [
            (5, 5, 3, 3), (12, 8, 4, 2), (22, 3, 2, 7),
            (18, 14, 6, 3), (5, 15, 3, 2)
        ]
        for gx, gy, gw, gh in gravestones:
            for x in range(gx, gx + gw):
                for y in range(gy, gy + gh):
                    if 0 < x < self.GRID_WIDTH -1 and 0 < y < self.GRID_HEIGHT -1:
                        self.walls.add((x, y))

        self.ritual_pos = (self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2)
        if self.ritual_pos in self.walls:
            self.walls.remove(self.ritual_pos)


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.particles = []
        self.items_collected = [False] * self.NUM_ITEMS

        # Get all valid spawn points
        valid_spawns = []
        for x in range(1, self.GRID_WIDTH - 1):
            for y in range(1, self.GRID_HEIGHT - 1):
                if (x, y) not in self.walls and (x, y) != self.ritual_pos:
                    valid_spawns.append((x, y))

        # Place entities
        spawn_indices = self.np_random.choice(len(valid_spawns), size=1 + self.NUM_ITEMS + self.NUM_GHOSTS, replace=False)
        spawn_points = [valid_spawns[i] for i in spawn_indices]

        self.player_pos = spawn_points.pop()
        
        self.items = []
        for i in range(self.NUM_ITEMS):
            self.items.append({"pos": spawn_points.pop(), "color": self.ITEM_COLORS[i]})

        self.ghosts = []
        # Ghost 1: Patrols a rectangle
        g1_pos = spawn_points.pop()
        self.ghosts.append({
            "pos": g1_pos,
            "path": [(3, 3), (28, 3), (28, 16), (3, 16)],
            "target_idx": 0
        })
        # Ghost 2: Patrols a corridor
        g2_pos = spawn_points.pop()
        self.ghosts.append({
            "pos": g2_pos,
            "path": [(15, 2), (15, 17)],
            "target_idx": 0
        })

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        space_held = action[1] == 1
        
        reward = 0
        self.steps += 1

        # 1. Player Movement
        moved = False
        if movement != 0:
            px, py = self.player_pos
            if movement == 1: new_pos = (px, py - 1) # Up
            elif movement == 2: new_pos = (px, py + 1) # Down
            elif movement == 3: new_pos = (px - 1, py) # Left
            else: new_pos = (px + 1, py) # Right

            if new_pos not in self.walls:
                self.player_pos = new_pos
                reward -= 0.01 # Small cost for taking a step
                moved = True
        
        # 2. Ghost Movement
        for ghost in self.ghosts:
            self._move_ghost(ghost)

        # 3. Interactions and Rewards
        # Item collection
        for i, item in enumerate(self.items):
            if not self.items_collected[i] and self.player_pos == item["pos"]:
                self.items_collected[i] = True
                self.score += 10
                reward += 10
                self._create_particles(item["pos"], item["color"], 20)
                # Sound: sfx_item_collect.wav

        # Ritual action
        if space_held and self.player_pos == self.ritual_pos:
            if all(self.items_collected):
                self.score += 100
                reward += 100
                self.game_over = True
                self._create_particles(self.ritual_pos, self.COLOR_RITUAL, 100, life=60, big=True)
                # Sound: sfx_ritual_success.wav
            else:
                reward -= 1 # Penalty for trying ritual without all items
                self._create_particles(self.ritual_pos, (100, 100, 100), 10, life=15)
                # Sound: sfx_ritual_fail.wav

        # 4. Update Particles
        self._update_particles()

        # 5. Check Termination Conditions
        # Ghost collision
        for ghost in self.ghosts:
            if self.player_pos == ghost["pos"]:
                self.game_over = True
                reward -= 50 # Penalty for dying
                self._create_particles(self.player_pos, self.COLOR_PLAYER, 50, life=40)
                # Sound: sfx_player_death.wav
                break
        
        # Max steps
        if self.steps >= self.MAX_STEPS:
            self.game_over = True

        terminated = self.game_over
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _move_ghost(self, ghost):
        """Moves a ghost one step towards its current target waypoint."""
        if not ghost["path"]: return

        target_pos = ghost["path"][ghost["target_idx"]]
        if ghost["pos"] == target_pos:
            ghost["target_idx"] = (ghost["target_idx"] + 1) % len(ghost["path"])
            target_pos = ghost["path"][ghost["target_idx"]]

        gx, gy = ghost["pos"]
        tx, ty = target_pos
        
        dx = tx - gx
        dy = ty - gy

        if abs(dx) > abs(dy):
            ghost["pos"] = (gx + np.sign(dx), gy)
        elif dy != 0:
            ghost["pos"] = (gx, gy + np.sign(dy))


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
            "items_collected": sum(self.items_collected),
        }

    def _render_game(self):
        # Render Walls
        for x, y in self.walls:
            rect = pygame.Rect(x * self.CELL_SIZE, y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_WALL, rect)

        # Render Ritual Site
        rx, ry = self.ritual_pos
        center_x = int((rx + 0.5) * self.CELL_SIZE)
        center_y = int((ry + 0.5) * self.CELL_SIZE)
        pulse = abs(math.sin(self.steps * 0.1))
        radius = int(self.CELL_SIZE * 0.4)
        
        glow_radius = int(radius + 4 + pulse * 4)
        glow_alpha = 40 + pulse * 20
        pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, glow_radius, (*self.COLOR_RITUAL, glow_alpha))
        pygame.gfxdraw.aacircle(self.screen, center_x, center_y, glow_radius, (*self.COLOR_RITUAL, glow_alpha))
        pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, self.COLOR_RITUAL)
        pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, self.COLOR_RITUAL)

        # Render Items
        for i, item in enumerate(self.items):
            if not self.items_collected[i]:
                ix, iy = item["pos"]
                center_x = int((ix + 0.5) * self.CELL_SIZE)
                center_y = int((iy + 0.5) * self.CELL_SIZE)
                pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, int(self.CELL_SIZE * 0.3), item["color"])
                pygame.gfxdraw.aacircle(self.screen, center_x, center_y, int(self.CELL_SIZE * 0.3), item["color"])

        # Render Particles
        for p in self.particles:
            p_x, p_y = p["pos"]
            alpha = int(255 * (p["life"] / p["max_life"]))
            color = (*p["color"], alpha)
            pygame.gfxdraw.filled_circle(self.screen, int(p_x), int(p_y), int(p["size"]), color)

        # Render Ghosts
        for ghost in self.ghosts:
            gx, gy = ghost["pos"]
            rect = pygame.Rect(gx * self.CELL_SIZE, gy * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
            flicker = self.np_random.integers(0, 2)
            color = self.COLOR_GHOST_1 if flicker == 0 else self.COLOR_GHOST_2
            pygame.draw.rect(self.screen, color, rect)

        # Render Player
        if not self.game_over or not any(self.player_pos == g['pos'] for g in self.ghosts):
            px, py = self.player_pos
            rect = pygame.Rect(px * self.CELL_SIZE, py * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_PLAYER, rect)


    def _render_ui(self):
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (1.5 * self.CELL_SIZE, 1.5 * self.CELL_SIZE))

        items_text = self.font_ui.render(f"ITEMS: {sum(self.items_collected)}/{self.NUM_ITEMS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(items_text, (self.SCREEN_WIDTH - 9 * self.CELL_SIZE, 1.5 * self.CELL_SIZE))

    def _create_particles(self, pos, color, count, life=20, big=False):
        center_x = (pos[0] + 0.5) * self.CELL_SIZE
        center_y = (pos[1] + 0.5) * self.CELL_SIZE
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4) if not big else self.np_random.uniform(2, 8)
            self.particles.append({
                "pos": [center_x, center_y],
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "size": self.np_random.integers(2, 5) if not big else self.np_random.integers(4, 9),
                "life": self.np_random.integers(life // 2, life),
                "max_life": life,
                "color": color
            })

    def _update_particles(self):
        active_particles = []
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][0] *= 0.95 # friction
            p["vel"][1] *= 0.95 # friction
            p["life"] -= 1
            p["size"] = max(0, p["size"] * 0.98)
            if p["life"] > 0:
                active_particles.append(p)
        self.particles = active_particles

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space
        test_obs, _ = self.reset()
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # To play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Haunted Graveyard")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    terminated = False
    
    print("\n" + "="*30)
    print("HAUNTED GRAVEYARD")
    print(env.user_guide)
    print("="*30 + "\n")

    while running:
        # --- Create Action from Human Input ---
        movement = 0 # no-op
        space_held = 0
        shift_held = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
        
        action = [movement, space_held, shift_held]
        
        # --- Step the Environment ---
        # Only step if an action is taken or if the game is over
        if any(action) or terminated:
            if terminated:
                print(f"Game Over! Final Score: {info['score']}. Resetting in 3 seconds...")
                pygame.time.wait(3000)
                obs, info = env.reset()
                terminated = False
            else:
                obs, reward, terminated, truncated, info = env.step(action)
        else: # If no action, just re-render the current state
             obs = env._get_observation()

        # --- Render to Screen ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit FPS for human play

    env.close()