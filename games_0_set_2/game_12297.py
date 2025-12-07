import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:41:46.607085
# Source Brief: brief_02297.md
# Brief Index: 2297
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the agent plays as a Pharaoh building a pyramid.
    
    The agent controls a cursor on a grid-based desert. It can collect resources,
    build pyramid layers, claim a vital oasis, and manipulate time to shift
    sand dunes. The goal is to achieve the highest score by building the tallest
    pyramid and maintaining control of the oasis within a fixed number of turns.
    
    Visuals are a key focus, with a top-down view, vibrant colors, particle effects,
    and a clear UI to provide a polished and engaging experience.
    """
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = "As a Pharaoh, gather resources, build a grand pyramid, and control a vital oasis to achieve the highest score."
    user_guide = "Use arrow keys to move. Press space to build, gather, or claim the oasis. Press shift to use a time artifact to shift sand dunes."
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_W, self.GRID_H = 32, 20
        self.TILE_SIZE = 20
        self.MAX_STEPS = 500
        self.PYRAMID_BASE_POS = (self.GRID_W // 2, self.GRID_H - 4)
        self.OASIS_POS = (self.GRID_W // 5, self.GRID_H // 5)
        self.OASIS_SIZE = 3
        
        self.MAX_RESOURCES = 100
        self.INITIAL_RESOURCES = 20
        self.PYRAMID_BUILD_COST = 5
        self.TIME_ARTIFACT_COST = 10
        self.RESOURCE_GAIN = 5
        self.OASIS_BONUS_RESOURCES = 1

        # --- Visuals ---
        self.COLOR_BG = (210, 180, 140)  # Desert Sand
        self.COLOR_SAND_DUNE = (194, 163, 121)
        self.COLOR_GRID = (189, 158, 119)
        self.COLOR_OASIS = (64, 164, 223)
        self.COLOR_PYRAMID = (139, 69, 19) # Dark Brown
        self.COLOR_RESOURCE = (50, 205, 50) # Lime Green
        self.COLOR_CURSOR = (255, 215, 0, 150) # Gold, semi-transparent
        self.COLOR_CURSOR_BORDER = (255, 215, 0)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_TEXT_SHADOW = (0, 0, 0)
        self.COLOR_TIME_EFFECT = (255, 0, 0)
        self.COLOR_OASIS_FLAG = (255, 99, 71) # Tomato Red
        
        # --- Gymnasium Spaces ---
        self.observation_space = Box(low=0, high=255, shape=(400, 640, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_tile = pygame.font.Font(None, 18)
        
        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.resources = 0
        self.pyramid_height = 0
        self.oasis_controlled = False
        self.cursor_pos = [0, 0]
        self.last_move_direction = 1  # 1:up, 2:down, 3:left, 4:right
        self.dunes = set()
        self.resource_tiles = set()
        self.particles = []
        
        # self.reset() is called by the wrapper, no need to call it here.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.resources = self.INITIAL_RESOURCES
        self.pyramid_height = 0
        self.oasis_controlled = False
        self.cursor_pos = [self.GRID_W // 2, self.GRID_H // 2]
        self.last_move_direction = 1
        self.particles = []
        
        self._initialize_grid()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1
        reward = 0

        self._move_cursor(movement)

        if shift_pressed:
            reward += self._use_time_artifact()
        elif space_pressed:
            reward += self._perform_contextual_action()

        if self.oasis_controlled:
            self.resources = min(self.MAX_RESOURCES, self.resources + self.OASIS_BONUS_RESOURCES)
            reward += 0.05 # Small reward for maintaining control

        self._update_particles()

        self.steps += 1
        self.score += reward
        
        terminated = self.steps >= self.MAX_STEPS
        if terminated:
            if self.pyramid_height > 0:
                reward += 50
                self.score += 50
            if self.oasis_controlled:
                reward += 25
                self.score += 25
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _initialize_grid(self):
        self.dunes = set()
        self.resource_tiles = set()
        for x in range(self.GRID_W):
            for y in range(self.GRID_H):
                is_oasis = self.OASIS_POS[0] <= x < self.OASIS_POS[0] + self.OASIS_SIZE and \
                           self.OASIS_POS[1] <= y < self.OASIS_POS[1] + self.OASIS_SIZE
                is_pyramid_base = (x, y) == self.PYRAMID_BASE_POS

                if not is_oasis and not is_pyramid_base:
                    if self.np_random.random() < 0.1:
                        self.dunes.add((x, y))
                    elif self.np_random.random() < 0.15:
                        self.resource_tiles.add((x, y))

    def _move_cursor(self, movement):
        if movement != 0:
            self.last_move_direction = movement
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        elif movement == 2: self.cursor_pos[1] += 1 # Down
        elif movement == 3: self.cursor_pos[0] -= 1 # Left
        elif movement == 4: self.cursor_pos[0] += 1 # Right
        
        self.cursor_pos[0] %= self.GRID_W
        self.cursor_pos[1] %= self.GRID_H

    def _use_time_artifact(self):
        if self.resources >= self.TIME_ARTIFACT_COST:
            self.resources -= self.TIME_ARTIFACT_COST
            
            dx, dy = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}.get(self.last_move_direction, (0, 0))
            
            new_dunes = set()
            for (x, y) in self.dunes:
                new_dunes.add(((x + dx) % self.GRID_W, (y + dy) % self.GRID_H))
            self.dunes = new_dunes
            
            self._create_effect(self.WIDTH / 2, self.HEIGHT / 2, 'time_warp')
            return -0.1  # Small penalty for a powerful action
        return 0

    def _perform_contextual_action(self):
        cursor = tuple(self.cursor_pos)
        
        if cursor == self.PYRAMID_BASE_POS and self.resources >= self.PYRAMID_BUILD_COST:
            self.resources -= self.PYRAMID_BUILD_COST
            self.pyramid_height += 1
            self._create_particles(
                (cursor[0] + 0.5) * self.TILE_SIZE, (cursor[1] + 0.5) * self.TILE_SIZE,
                15, self.COLOR_PYRAMID, life=30
            )
            return 0.1 * self.pyramid_height

        is_oasis = self.OASIS_POS[0] <= cursor[0] < self.OASIS_POS[0] + self.OASIS_SIZE and \
                   self.OASIS_POS[1] <= cursor[1] < self.OASIS_POS[1] + self.OASIS_SIZE
        if is_oasis and not self.oasis_controlled:
            self.oasis_controlled = True
            self._create_particles(
                (cursor[0] + 0.5) * self.TILE_SIZE, (cursor[1] + 0.5) * self.TILE_SIZE,
                30, self.COLOR_OASIS, life=40, speed=3
            )
            return 5.0

        if cursor in self.resource_tiles and cursor not in self.dunes:
            self.resource_tiles.remove(cursor)
            self.resources = min(self.MAX_RESOURCES, self.resources + self.RESOURCE_GAIN)
            self._create_particles(
                (cursor[0] + 0.5) * self.TILE_SIZE, (cursor[1] + 0.5) * self.TILE_SIZE,
                10, self.COLOR_RESOURCE, life=20, speed=2
            )
            return 1.0
        
        return 0

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
            "resources": self.resources,
            "pyramid_height": self.pyramid_height,
            "oasis_controlled": self.oasis_controlled,
        }

    def _render_game(self):
        # Draw Grid
        for x in range(self.GRID_W):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x * self.TILE_SIZE, 0), (x * self.TILE_SIZE, self.HEIGHT))
        for y in range(self.GRID_H):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y * self.TILE_SIZE), (self.WIDTH, y * self.TILE_SIZE))

        # Draw Oasis
        oasis_rect = pygame.Rect(self.OASIS_POS[0] * self.TILE_SIZE, self.OASIS_POS[1] * self.TILE_SIZE,
                                 self.OASIS_SIZE * self.TILE_SIZE, self.OASIS_SIZE * self.TILE_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_OASIS, oasis_rect)
        if self.oasis_controlled:
            flag_pos = (oasis_rect.centerx, oasis_rect.centery)
            pygame.draw.circle(self.screen, self.COLOR_OASIS_FLAG, flag_pos, 5)
            pygame.draw.line(self.screen, self.COLOR_OASIS_FLAG, flag_pos, (flag_pos[0], flag_pos[1] - 10), 2)

        # Draw Resources
        for (x, y) in self.resource_tiles:
            if (x, y) not in self.dunes:
                center = (x * self.TILE_SIZE + self.TILE_SIZE / 2, y * self.TILE_SIZE + self.TILE_SIZE / 2)
                pts = [(center[0], center[1] - 6), (center[0] + 6, center[1]), (center[0], center[1] + 6), (center[0] - 6, center[1])]
                pygame.draw.polygon(self.screen, self.COLOR_RESOURCE, pts)
                pygame.draw.polygon(self.screen, (255, 255, 255), pts, 1)

        # Draw Pyramid
        px, py = self.PYRAMID_BASE_POS
        base_rect = pygame.Rect(px * self.TILE_SIZE, py * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_PYRAMID, base_rect)
        for i in range(self.pyramid_height):
            lerp = min(1.0, i / 15.0)
            color = tuple(int(self.COLOR_PYRAMID[c] + lerp * (self.COLOR_BG[c] - self.COLOR_PYRAMID[c])) for c in range(3))
            p_rect = base_rect.inflate(- (i + 1) * 2, - (i + 1) * 2)
            if p_rect.width > 0:
                pygame.draw.rect(self.screen, color, p_rect)
        if self.pyramid_height > 0:
            self._draw_text_shadow(str(self.pyramid_height), self.font_tile, self.COLOR_TEXT, base_rect.center, center=True)

        # Draw Dunes
        for (x, y) in self.dunes:
            rect = pygame.Rect(x * self.TILE_SIZE, y * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_SAND_DUNE, rect)
            for i in range(3):
                y_offset = 5 + i * 5
                start_pos = (rect.left, rect.top + y_offset + math.sin(self.steps / 5 + x) * 2)
                end_pos = (rect.right, rect.top + y_offset + math.cos(self.steps / 5 + y) * 2)
                pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos)

        self._render_particles()

        # Draw Cursor
        cursor_rect = pygame.Rect(self.cursor_pos[0] * self.TILE_SIZE, self.cursor_pos[1] * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
        s = pygame.Surface((self.TILE_SIZE, self.TILE_SIZE), pygame.SRCALPHA)
        pygame.draw.rect(s, self.COLOR_CURSOR, s.get_rect())
        self.screen.blit(s, cursor_rect.topleft)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR_BORDER, cursor_rect, 2)

    def _render_ui(self):
        # UI Background Panel
        panel_rect = pygame.Rect(0, 0, self.WIDTH, 40)
        s = pygame.Surface((self.WIDTH, 40), pygame.SRCALPHA)
        s.fill((0, 0, 0, 100))
        self.screen.blit(s, (0, 0))

        # Resources
        res_text = f"Resources: {self.resources}"
        self._draw_text_shadow(res_text, self.font_ui, self.COLOR_TEXT, (10, 10))
        # Pyramid Height
        pyr_text = f"Pyramid: {self.pyramid_height}"
        self._draw_text_shadow(pyr_text, self.font_ui, self.COLOR_TEXT, (180, 10))
        # Oasis Status
        oasis_color = (144, 238, 144) if self.oasis_controlled else (255, 100, 100)
        oasis_text = "Oasis: Controlled" if self.oasis_controlled else "Oasis: Free"
        self._draw_text_shadow(oasis_text, self.font_ui, oasis_color, (330, 10))
        # Turn
        turn_text = f"Turn: {self.steps}/{self.MAX_STEPS}"
        self._draw_text_shadow(turn_text, self.font_ui, self.COLOR_TEXT, (self.WIDTH - 130, 10))

    def _draw_text_shadow(self, text, font, color, pos, center=False):
        text_surf = font.render(text, True, color)
        shadow_surf = font.render(text, True, self.COLOR_TEXT_SHADOW)
        text_rect = text_surf.get_rect()
        if center:
            text_rect.center = pos
        else:
            text_rect.topleft = pos
        
        self.screen.blit(shadow_surf, (text_rect.x + 1, text_rect.y + 1))
        self.screen.blit(text_surf, text_rect)

    def _create_particles(self, x, y, count=10, color=(255, 255, 255), life=20, speed=2):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            s = self.np_random.uniform(0.5, speed)
            self.particles.append({
                'x': x, 'y': y, 'vx': math.cos(angle) * s, 'vy': math.sin(angle) * s,
                'life': life, 'max_life': life, 'color': color, 'type': 'spark'
            })

    def _create_effect(self, x, y, effect_type):
        if effect_type == 'time_warp':
            self.particles.append({
                'x': x, 'y': y, 'life': 20, 'max_life': 20,
                'radius': 0, 'type': 'time_warp'
            })

    def _update_particles(self):
        for p in self.particles:
            p['life'] -= 1
            if p['type'] == 'spark':
                p['x'] += p['vx']
                p['y'] += p['vy']
                p['vy'] += 0.1  # Gravity
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _render_particles(self):
        for p in self.particles:
            alpha_factor = p['life'] / p['max_life']
            if p['type'] == 'spark':
                size = int(3 * alpha_factor)
                if size > 0:
                    pygame.draw.circle(self.screen, p['color'], (int(p['x']), int(p['y'])), size)
            elif p['type'] == 'time_warp':
                color = self.COLOR_TIME_EFFECT + (int(150 * alpha_factor),)
                radius = int((1 - alpha_factor) * self.WIDTH / 2)
                if radius > 1:
                    pygame.gfxdraw.aacircle(self.screen, int(p['x']), int(p['y']), radius, color)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc is False
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It will not run in a headless environment.
    os.environ["SDL_VIDEODRIVER"] = "x11"
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    screen = pygame.display.set_mode((640, 400))
    pygame.display.set_caption("Pharaoh's Gambit")
    clock = pygame.time.Clock()

    total_reward = 0
    
    print(GameEnv.user_guide)
    print("Goal: Build pyramid, control oasis. Game ends after 500 turns.")

    while not done:
        # --- Manual Control ---
        movement, space, shift = 0, 0, 0
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
        
        # --- Step Environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        # --- Render ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(10) # Control game speed for manual play

    print(f"Game Over! Final Info: {info}")
    print(f"Total Reward: {total_reward}")
    env.close()