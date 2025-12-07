
# Generated: 2025-08-28T04:56:47.260356
# Source Brief: brief_02461.md
# Brief Index: 2461

        
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
        "Controls: Use arrow keys to move. Press Space to place a gem on a target."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Collect colored gems and place them on their matching targets before time runs out."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.GRID_WIDTH = 16
        self.GRID_HEIGHT = 10
        self.CELL_SIZE = self.SCREEN_WIDTH // self.GRID_WIDTH
        self.MAX_STEPS = 6000
        self.NUM_GEMS = 3

        # --- Colors ---
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_GRID = (40, 40, 50)
        self.COLOR_PLAYER = (60, 160, 255)
        self.COLOR_PLAYER_OUTLINE = (180, 220, 255)
        self.GEM_COLORS = [
            (255, 70, 70),  # Red
            (70, 255, 70),  # Green
            (255, 255, 70), # Yellow
        ]
        self.UI_TEXT_COLOR = (220, 220, 220)

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_big = pygame.font.SysFont("Verdana", 48, bold=True)

        # --- Game State ---
        self.player_pos = None
        self.gems = []
        self.targets = []
        self.held_gem_idx = -1
        self.steps = 0
        self.score = 0
        self.time_remaining = 0
        self.game_over_message = ""
        self.particles = []

        self._initialize_state() # Initial call to setup state for validation
        self.validate_implementation()

    def _initialize_state(self):
        """Initializes all game state variables for a new episode."""
        self.steps = 0
        self.score = 0
        self.time_remaining = self.MAX_STEPS
        self.game_over_message = ""
        self.particles = []

        # Generate unique positions for player, gems, and targets
        all_pos = set()
        
        def get_random_pos():
            while True:
                pos = (
                    self.np_random.integers(0, self.GRID_WIDTH),
                    self.np_random.integers(0, self.GRID_HEIGHT)
                )
                if pos not in all_pos:
                    all_pos.add(pos)
                    return pos

        self.player_pos = get_random_pos()

        self.gems = []
        gem_colors_shuffled = self.GEM_COLORS[:]
        self.np_random.shuffle(gem_colors_shuffled)
        for i in range(self.NUM_GEMS):
            self.gems.append({
                "pos": get_random_pos(),
                "color": gem_colors_shuffled[i],
                "id": i,
                "status": "available" # available, held, placed
            })

        self.targets = []
        # Ensure target colors match gem colors for solvability
        for i in range(self.NUM_GEMS):
             self.targets.append({
                "pos": get_random_pos(),
                "color": self.gems[i]['color'],
                "id": i # Matches gem id
            })

        self.held_gem_idx = -1

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._initialize_state()
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        terminated = False
        
        movement, space_pressed, _ = action[0], action[1] == 1, action[2] == 1
        
        # Game logic proceeds only if not in a terminal state
        if self.game_over_message == "":
            dist_before = self._get_dist_to_nearest_gem()

            # --- Apply Movement ---
            dx, dy = 0, 0
            if movement == 1: dy = -1  # Up
            elif movement == 2: dy = 1   # Down
            elif movement == 3: dx = -1  # Left
            elif movement == 4: dx = 1   # Right
            
            new_x = max(0, min(self.GRID_WIDTH - 1, self.player_pos[0] + dx))
            new_y = max(0, min(self.GRID_HEIGHT - 1, self.player_pos[1] + dy))
            self.player_pos = (new_x, new_y)

            # --- Calculate Movement Reward ---
            dist_after = self._get_dist_to_nearest_gem()
            if dist_before is not None and dist_after is not None:
                if dist_after < dist_before:
                    reward += 0.1
                elif dist_after > dist_before:
                    reward -= 0.01
            
            # --- Game Logic: Collection and Placement ---
            # Gem Collection
            if self.held_gem_idx == -1:
                for i, gem in enumerate(self.gems):
                    if gem["status"] == "available" and gem["pos"] == self.player_pos:
                        gem["status"] = "held"
                        self.held_gem_idx = i
                        reward += 1.0
                        self._create_particles(self.player_pos, gem["color"], 20)
                        # sound: gem_collect.wav
                        break
            
            # Gem Placement
            if space_pressed and self.held_gem_idx != -1:
                placed_correctly = False
                on_any_target = False
                held_gem = self.gems[self.held_gem_idx]

                for target in self.targets:
                    if target["pos"] == self.player_pos:
                        on_any_target = True
                        if target["color"] == held_gem["color"]:
                            held_gem["status"] = "placed"
                            self.held_gem_idx = -1
                            reward += 5.0
                            placed_correctly = True
                            self._create_particles(self.player_pos, target["color"], 40, is_ring=True)
                            # sound: place_correct.wav
                            break
                
                if on_any_target and not placed_correctly:
                    reward -= 1.0 # Penalty for wrong target
                    # sound: place_wrong.wav

        # --- Update State & Check Termination ---
        self.steps += 1
        self.time_remaining -= 1
        self._update_particles()
        
        if all(gem["status"] == "placed" for gem in self.gems):
            if self.game_over_message == "": # Add reward only on first frame of win
                reward += 100.0
                self.game_over_message = "SUCCESS!"
            terminated = True

        if self.time_remaining <= 0 and not terminated:
            if self.game_over_message == "": # Add penalty only on first frame of loss
                reward -= 100.0
                self.game_over_message = "TIME'S UP!"
            terminated = True

        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _get_dist_to_nearest_gem(self):
        if self.held_gem_idx != -1:
            return None
        
        available_gems = [g for g in self.gems if g["status"] == "available"]
        if not available_gems:
            return None
        
        player_x, player_y = self.player_pos
        min_dist = float('inf')
        for gem in available_gems:
            dist = abs(player_x - gem["pos"][0]) + abs(player_y - gem["pos"][1])
            min_dist = min(min_dist, dist)
        return min_dist

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        if self.game_over_message:
            self._render_game_over()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        gems_placed = sum(1 for gem in self.gems if gem["status"] == "placed")
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining,
            "gems_placed": gems_placed,
            "gems_total": self.NUM_GEMS,
        }

    def _grid_to_screen(self, pos):
        x = pos[0] * self.CELL_SIZE + self.CELL_SIZE // 2
        y = pos[1] * self.CELL_SIZE + self.CELL_SIZE // 2
        return int(x), int(y)

    def _render_game(self):
        for x in range(0, self.SCREEN_WIDTH, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

        for target in self.targets:
            center_pos = self._grid_to_screen(target["pos"])
            radius = self.CELL_SIZE // 3
            is_fulfilled = any(g['status'] == 'placed' and g['color'] == target['color'] for g in self.gems)
            
            if is_fulfilled:
                pygame.gfxdraw.filled_circle(self.screen, center_pos[0], center_pos[1], radius, target["color"])
                pygame.gfxdraw.aacircle(self.screen, center_pos[0], center_pos[1], radius, target["color"])
            else:
                pygame.gfxdraw.aacircle(self.screen, center_pos[0], center_pos[1], radius, target["color"])
                pygame.gfxdraw.aacircle(self.screen, center_pos[0], center_pos[1], radius - 1, target["color"])

        for gem in self.gems:
            if gem["status"] == "available":
                center_pos = self._grid_to_screen(gem["pos"])
                radius = self.CELL_SIZE // 4
                pygame.gfxdraw.filled_circle(self.screen, center_pos[0], center_pos[1], radius, gem["color"])
                pygame.gfxdraw.aacircle(self.screen, center_pos[0], center_pos[1], radius, gem["color"])

        self._render_particles()

        player_screen_pos = self._grid_to_screen(self.player_pos)
        player_rect = pygame.Rect(player_screen_pos[0] - self.CELL_SIZE // 3, player_screen_pos[1] - self.CELL_SIZE // 3, self.CELL_SIZE * 2 // 3, self.CELL_SIZE * 2 // 3)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=4)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER_OUTLINE, player_rect, width=2, border_radius=4)

        if self.held_gem_idx != -1:
            gem = self.gems[self.held_gem_idx]
            gem_rect = pygame.Rect(player_screen_pos[0] - self.CELL_SIZE // 6, player_screen_pos[1] - self.CELL_SIZE // 6, self.CELL_SIZE // 3, self.CELL_SIZE // 3)
            pygame.draw.rect(self.screen, gem["color"], gem_rect, border_radius=2)
            
    def _render_ui(self):
        gems_placed = sum(1 for gem in self.gems if gem["status"] == "placed")
        gems_text = f"Gems: {gems_placed}/{self.NUM_GEMS}"
        gems_surf = self.font_ui.render(gems_text, True, self.UI_TEXT_COLOR)
        self.screen.blit(gems_surf, (10, 10))

        time_text = f"Time: {self.time_remaining}"
        time_surf = self.font_ui.render(time_text, True, self.UI_TEXT_COLOR)
        self.screen.blit(time_surf, (self.SCREEN_WIDTH - time_surf.get_width() - 10, 10))
        
    def _render_game_over(self):
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        text_surf = self.font_big.render(self.game_over_message, True, self.UI_TEXT_COLOR)
        text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
        self.screen.blit(overlay, (0, 0))
        self.screen.blit(text_surf, text_rect)

    def _create_particles(self, grid_pos, color, count, is_ring=False):
        screen_pos = self._grid_to_screen(grid_pos)
        for _ in range(count):
            if is_ring:
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(2, 4)
                vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            else:
                vel = [self.np_random.uniform(-3, 3), self.np_random.uniform(-3, 3)]
            self.particles.append({"pos": list(screen_pos), "vel": vel, "lifespan": self.np_random.integers(20, 40), "color": color, "radius": self.np_random.uniform(2, 5)})

    def _update_particles(self):
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["lifespan"] -= 1
        self.particles = [p for p in self.particles if p["lifespan"] > 0]

    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p["lifespan"] / 40))))
            color = (*p["color"], alpha)
            pos = (int(p["pos"][0]), int(p["pos"][1]))
            radius = int(p["radius"] * (p["lifespan"] / 40))
            if radius > 0:
                temp_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (radius, radius), radius)
                self.screen.blit(temp_surf, (pos[0] - radius, pos[1] - radius))

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
        assert trunc == False
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    terminated = False
    
    key_map = {pygame.K_UP: 1, pygame.K_DOWN: 2, pygame.K_LEFT: 3, pygame.K_RIGHT: 4}

    pygame.display.set_caption("Gem Collector")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    total_reward = 0
    running = True
    
    while running:
        movement_action = 0
        space_action = 0
        shift_action = 0
        action_taken = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                action_taken = True
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key in key_map:
                    movement_action = key_map[event.key]
                if event.key == pygame.K_SPACE:
                    space_action = 1
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                    shift_action = 1
                if event.key == pygame.K_r: # Reset on 'r'
                    obs, info = env.reset()
                    total_reward = 0
                    terminated = False
                    action_taken = False # Don't step on reset
        
        if not terminated and action_taken:
            action = [movement_action, space_action, shift_action]
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            print(f"Step: {info['steps']}, Action: {action}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}, Terminated: {terminated}")

        screen.blit(env.screen, (0, 0))
        pygame.display.flip()
        env.clock.tick(60)
        
    env.close()