import gymnasium as gym
import os
import pygame
import numpy as np
import math
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
pygame.init()
pygame.font.init()

# The following import is only needed for the visual rendering in the __main__ block
# and is not strictly required by the environment itself.
import pygame.gfxdraw


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}
    game_description = (
        "A 2D platformer where you grow platforms by matching roots and plant trees to restore the world. "
        "Manipulate the water level to access new areas."
    )
    user_guide = (
        "Controls: Use ←→ arrow keys to move and ↑ to jump. "
        "Press space to interact with roots or plant trees. Press shift to switch the water level."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.WORLD_WIDTH = 1920
        self.FPS = 30
        self.MAX_STEPS = 2500

        # Colors
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_PLAYER = (50, 255, 50)
        self.COLOR_PLAYER_GLOW = (150, 255, 150)
        self.COLOR_STATIC_PLATFORM = (80, 50, 20)
        self.COLOR_GROWN_PLATFORM = (100, 180, 80)
        self.COLOR_WATER = (50, 100, 200)
        self.COLOR_ROOT_RED = (255, 50, 50)
        self.COLOR_ROOT_YELLOW = (255, 255, 50)
        self.COLOR_ROOT_PURPLE = (180, 50, 255)
        self.COLOR_UI_TEXT = (255, 255, 255)
        self.COLOR_FERTILE_SOIL = (100, 80, 60)
        self.COLOR_TREE_TRUNK = (139, 69, 19)
        self.COLOR_TREE_LEAVES = (34, 139, 34)

        # Physics
        self.GRAVITY = 0.8
        self.PLAYER_SPEED = 5.0
        self.JUMP_STRENGTH = -12.0
        self.FRICTION = 0.85
        self.MAX_FALL_SPEED = 15.0
        self.PLAYER_SIZE = 20

        # Game Mechanics
        self.INTERACTION_RADIUS = 40
        self.WATER_LEVEL_HIGH = self.SCREEN_HEIGHT - 100
        self.WATER_LEVEL_LOW = self.SCREEN_HEIGHT + 50  # Off-screen

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = gym.spaces.MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_ui = pygame.font.Font(None, 24)
            self.font_big = pygame.font.Font(None, 50)
        except IOError:
            self.font_ui = pygame.font.SysFont("sans-serif", 24)
            self.font_big = pygame.font.SysFont("sans-serif", 50)

        # --- State Variables (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_rect = None
        self.player_vel = None
        self.is_on_ground = False
        self.camera_x = 0.0
        self.target_camera_x = 0.0
        self.water_level_state = 'low'
        self.static_platforms = []
        self.grown_platforms = []
        self.roots = []
        self.tree_spots = []
        self.planted_trees = []
        self.selected_root_idx = -1
        self.last_space_held = False
        self.last_shift_held = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False

        # Player state
        self.player_rect = pygame.Rect(100, 200, self.PLAYER_SIZE, self.PLAYER_SIZE)
        self.player_vel = pygame.Vector2(0, 0)
        self.is_on_ground = False

        # World state
        self.camera_x = 0.0
        self.target_camera_x = 0.0
        self.water_level_state = 'low'

        # Level definition
        self._define_level()

        # Input state
        self.last_space_held = False
        self.last_shift_held = False
        self.selected_root_idx = -1

        return self._get_observation(), self._get_info()

    def _define_level(self):
        self.static_platforms = [
            # Ground
            pygame.Rect(0, 380, self.WORLD_WIDTH, 20),
            # Starting area platforms
            pygame.Rect(200, 320, 150, 20),
            pygame.Rect(50, 250, 100, 20),
            # Area after first bridge
            pygame.Rect(850, 380, 250, 20),
            # High wall
            pygame.Rect(1100, 280, 20, 100),
            # Area after high wall
            pygame.Rect(1120, 380, 300, 20),
            pygame.Rect(1250, 300, 150, 20),
            # Final area
            pygame.Rect(1650, 380, 270, 20)
        ]
        self.grown_platforms = []

        self.roots = [
            # Red pair for first bridge
            {'pos': (550, 370), 'color': self.COLOR_ROOT_RED, 'id': 0, 'pair_id': 1, 'active': False},
            {'pos': (900, 370), 'color': self.COLOR_ROOT_RED, 'id': 1, 'pair_id': 0, 'active': False},
            # Yellow pair for second bridge (needs low water)
            {'pos': (1450, 370), 'color': self.COLOR_ROOT_YELLOW, 'id': 2, 'pair_id': 3, 'active': False},
            {'pos': (1680, 370), 'color': self.COLOR_ROOT_YELLOW, 'id': 3, 'pair_id': 2, 'active': False},
            # Purple pair for high platform
            {'pos': (1150, 270), 'color': self.COLOR_ROOT_PURPLE, 'id': 4, 'pair_id': 5, 'active': False},
            {'pos': (1350, 290), 'color': self.COLOR_ROOT_PURPLE, 'id': 5, 'pair_id': 4, 'active': False},
        ]

        self.tree_spots = [
            pygame.Rect(250, 300, 40, 20),  # On a starting platform
            pygame.Rect(1300, 280, 40, 20),  # After high wall
            pygame.Rect(1800, 360, 40, 20),  # Final area
        ]
        self.planted_trees = []

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0.0

        # --- Action Handling ---
        movement = action[0]
        space_held = action[1] == 1
        shift_held = action[2] == 1

        space_pressed = space_held and not self.last_space_held
        shift_pressed = shift_held and not self.last_shift_held

        self.last_space_held = space_held
        self.last_shift_held = shift_held

        if shift_pressed:
            self.water_level_state = 'high' if self.water_level_state == 'low' else 'low'

        # --- Player Physics ---
        self.player_vel.y += self.GRAVITY
        self.player_vel.y = min(self.player_vel.y, self.MAX_FALL_SPEED)

        if movement == 3:  # Left
            self.player_vel.x = -self.PLAYER_SPEED
        elif movement == 4:  # Right
            self.player_vel.x = self.PLAYER_SPEED
        else:  # No horizontal movement, apply friction
            self.player_vel.x *= self.FRICTION

        if movement == 1 and self.is_on_ground:  # Up
            self.player_vel.y = self.JUMP_STRENGTH
            self.is_on_ground = False

        self.player_rect.x += int(self.player_vel.x)
        self.player_rect.y += int(self.player_vel.y)
        self.is_on_ground = False

        # --- Collisions & Interactions ---
        self.player_rect.left = max(0, self.player_rect.left)
        self.player_rect.right = min(self.WORLD_WIDTH, self.player_rect.right)

        all_platforms = self.static_platforms + self.grown_platforms
        for plat in all_platforms:
            if self.player_rect.colliderect(plat):
                if self.player_vel.y > 0 and self.player_rect.bottom - self.player_vel.y <= plat.top + 1:
                    self.player_rect.bottom = plat.top
                    self.player_vel.y = 0
                    self.is_on_ground = True
                elif self.player_vel.y < 0 and self.player_rect.top - self.player_vel.y >= plat.bottom - 1:
                    self.player_rect.top = plat.bottom
                    self.player_vel.y = 0
                elif self.player_vel.x > 0 and self.player_rect.right - self.player_vel.x <= plat.left + 1:
                    self.player_rect.right = plat.left
                    self.player_vel.x = 0
                elif self.player_vel.x < 0 and self.player_rect.left - self.player_vel.x >= plat.right - 1:
                    self.player_rect.left = plat.right
                    self.player_vel.x = 0

        current_water_y = self.WATER_LEVEL_HIGH if self.water_level_state == 'high' else self.WATER_LEVEL_LOW
        if self.player_rect.bottom > current_water_y:
            reward -= 0.1  # Penalty for being in water
            self.player_vel.y *= 0.9  # Buoyancy
            self.is_on_ground = True  # Can jump out of water

        if space_pressed:
            interaction_reward = self._handle_interaction()
            reward += interaction_reward

        # --- Camera Update ---
        self.target_camera_x = self.player_rect.centerx - self.SCREEN_WIDTH / 2
        self.target_camera_x = max(0, min(self.target_camera_x, self.WORLD_WIDTH - self.SCREEN_WIDTH))
        self.camera_x = self.camera_x * 0.9 + self.target_camera_x * 0.1

        # --- Termination and Score ---
        terminated = False
        truncated = False
        if self.player_rect.top > self.SCREEN_HEIGHT + 50:
            terminated = True
            reward -= 100.0
            self.game_over = True

        if len(self.planted_trees) == len(self.tree_spots):
            terminated = True
            reward += 100.0
            self.game_over = True

        if self.steps >= self.MAX_STEPS:
            truncated = True
            self.game_over = True

        self.score += reward

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_interaction(self):
        player_center = self.player_rect.center
        closest_root_idx, min_dist = -1, float('inf')
        for i, root in enumerate(self.roots):
            if root['active']: continue
            dist = math.hypot(player_center[0] - root['pos'][0], player_center[1] - root['pos'][1])
            if dist < min_dist:
                min_dist = dist
                closest_root_idx = i

        if closest_root_idx != -1 and min_dist < self.INTERACTION_RADIUS:
            if self.selected_root_idx == -1:
                self.selected_root_idx = closest_root_idx
            else:
                root1 = self.roots[self.selected_root_idx]
                root2 = self.roots[closest_root_idx]
                if root1['pair_id'] == root2['id'] and root1['color'] == root2['color']:
                    p1 = root1['pos']
                    p2 = root2['pos']
                    rect = pygame.Rect(min(p1[0], p2[0]), min(p1[1], p2[1]) - 5, abs(p1[0] - p2[0]), 10)
                    self.grown_platforms.append(rect)
                    root1['active'] = True
                    root2['active'] = True
                    self.selected_root_idx = -1
                    return 0.1
                else:
                    self.selected_root_idx = -1
            return 0.0

        for i, spot in enumerate(self.tree_spots):
            if i not in self.planted_trees and self.player_rect.colliderect(spot):
                self.planted_trees.append(i)
                return 1.0

        self.selected_root_idx = -1
        return 0.0

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        cam_x = int(self.camera_x)

        water_y = self.WATER_LEVEL_HIGH if self.water_level_state == 'high' else self.WATER_LEVEL_LOW
        if water_y < self.SCREEN_HEIGHT:
            water_surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT - water_y), pygame.SRCALPHA)
            water_surface.fill((*self.COLOR_WATER, 150))
            for x in range(self.SCREEN_WIDTH):
                y_offset = int(math.sin((x + self.steps * 2) * 0.05) * 3)
                pygame.draw.line(water_surface, (*self.COLOR_WATER, 200), (x, y_offset), (x, y_offset + 2), 2)
            self.screen.blit(water_surface, (0, water_y))

        for spot in self.tree_spots:
            pygame.draw.rect(self.screen, self.COLOR_FERTILE_SOIL, spot.move(-cam_x, 0))

        for tree_idx in self.planted_trees:
            spot = self.tree_spots[tree_idx]
            trunk = pygame.Rect(spot.centerx - 5 - cam_x, spot.top - 40, 10, 40)
            pygame.draw.rect(self.screen, self.COLOR_TREE_TRUNK, trunk)
            pygame.gfxdraw.filled_circle(self.screen, spot.centerx - cam_x, spot.top - 40, 20, self.COLOR_TREE_LEAVES)
            pygame.gfxdraw.aacircle(self.screen, spot.centerx - cam_x, spot.top - 40, 20, self.COLOR_TREE_LEAVES)

        for plat in self.static_platforms:
            pygame.draw.rect(self.screen, self.COLOR_STATIC_PLATFORM, plat.move(-cam_x, 0))
        for plat in self.grown_platforms:
            pygame.draw.rect(self.screen, self.COLOR_GROWN_PLATFORM, plat.move(-cam_x, 0))

        for i, root in enumerate(self.roots):
            if root['active']: continue
            pos = (root['pos'][0] - cam_x, root['pos'][1])
            if i == self.selected_root_idx:
                pulse_radius = 12 + int(math.sin(self.steps * 0.2) * 2)
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], pulse_radius, self.COLOR_UI_TEXT)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 8, root['color'])
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 8, root['color'])

        player_center = (self.player_rect.centerx - cam_x, self.player_rect.centery)
        glow_surf = pygame.Surface((self.PLAYER_SIZE * 3, self.PLAYER_SIZE * 3), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (*self.COLOR_PLAYER_GLOW, 50),
                           (self.PLAYER_SIZE * 1.5, self.PLAYER_SIZE * 1.5), self.PLAYER_SIZE * 1.5)
        self.screen.blit(glow_surf, (player_center[0] - self.PLAYER_SIZE * 1.5, player_center[1] - self.PLAYER_SIZE * 1.5))

        pygame.draw.rect(self.screen, self.COLOR_PLAYER, self.player_rect.move(-cam_x, 0))

    def _render_ui(self):
        score_text = self.font_ui.render(f"Score: {self.score:.1f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        tree_text = self.font_ui.render(f"Trees: {len(self.planted_trees)} / {len(self.tree_spots)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(tree_text, (self.SCREEN_WIDTH - tree_text.get_width() - 10, 10))

        bar_width, bar_height = 100, 15
        bar_x, bar_y = self.SCREEN_WIDTH // 2 - bar_width // 2, 10
        pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, (bar_x - 2, bar_y - 2, bar_width + 4, bar_height + 4), 1)

        fill_width = bar_width if self.water_level_state == 'high' else 0
        if fill_width > 0:
            pygame.draw.rect(self.screen, self.COLOR_WATER, (bar_x, bar_y, fill_width, bar_height))
        water_text = self.font_ui.render("Water", True, self.COLOR_UI_TEXT)
        self.screen.blit(water_text, (bar_x + bar_width + 5, bar_y))

        if self.game_over:
            msg = "VICTORY!" if len(self.planted_trees) == len(self.tree_spots) else "GAME OVER"
            end_text = self.font_big.render(msg, True, self.COLOR_UI_TEXT)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_pos": (self.player_rect.x, self.player_rect.y),
            "planted_trees": len(self.planted_trees),
            "water_level": self.water_level_state
        }

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually
    # It will not run in a headless environment.
    os.environ.pop("SDL_VIDEODRIVER", None)
    pygame.quit() # a bit of a hack to re-init with video
    pygame.init()

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()

    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Terraform Root Platformer")
    clock = pygame.time.Clock()

    done = False
    total_reward = 0

    while not done:
        movement, space_held, shift_held = 0, 0, 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            movement = 4

        if keys[pygame.K_UP] or keys[pygame.K_w]:
            movement = 1

        if keys[pygame.K_SPACE]:
            space_held = 1

        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_held = 1

        action = [movement, space_held, shift_held]

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print(f"Episode finished. Total reward: {total_reward:.2f}")
            print(f"Info: {info}")
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(env.FPS)

    env.close()