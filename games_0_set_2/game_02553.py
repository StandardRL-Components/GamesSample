
# Generated: 2025-08-28T05:13:34.871067
# Source Brief: brief_02553.md
# Brief Index: 2553

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
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
        "Controls: ↑/↓ to fly up and down. Avoid trees and fly through the gold rings."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Soar through a procedurally generated forest, navigating rings and dodging trees in this side-scrolling arcade flight game."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30

        # Colors
        self.COLOR_SKY_TOP = (112, 164, 255)
        self.COLOR_SKY_BOTTOM = (160, 200, 255)
        self.COLOR_PLAYER = (255, 50, 50)
        self.COLOR_RING = (255, 215, 0)
        self.COLOR_RING_COLLECTED = (150, 150, 150)
        self.COLOR_TREE_TRUNK = (139, 69, 19)
        self.COLOR_TREE_LEAVES = (34, 139, 34)
        self.COLOR_UI_TEXT = (255, 255, 255)
        self.COLOR_UI_SHADOW = (50, 50, 50)

        # Player Physics
        self.PLAYER_X_POS = 100
        self.PLAYER_SIZE = 12
        self.GRAVITY = 0.4
        self.LIFT = -1.2
        self.DIVE = 0.8
        self.MAX_VEL = 8

        # World
        self.SCROLL_SPEED = 5
        self.TOTAL_RINGS = 10
        self.MAX_TREES_HIT = 3
        self.MAX_STEPS = 1000
        self.RING_SPACING = 500
        self.RING_RADIUS = 30
        self.INITIAL_TREE_DENSITY = 0.0015 # trees per pixel
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        
        # Etc...        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_y = 0
        self.player_vy = 0
        self.rings_collected = 0
        self.trees_hit = 0
        self.rings = []
        self.trees = []
        self.particles = []
        self.world_generated_x = 0
        self.screen_shake = 0
        
        # Initialize state variables
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player_y = self.HEIGHT // 2
        self.player_vy = 0
        
        self.rings_collected = 0
        self.trees_hit = 0
        
        self.rings = []
        self.trees = []
        self.particles = []
        self.world_generated_x = 0
        self.screen_shake = 0

        # Procedural generation
        self._generate_world_chunk(self.WIDTH * 4) # Generate initial world
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def _generate_world_chunk(self, chunk_width):
        start_x = self.world_generated_x
        end_x = start_x + chunk_width
        
        # Generate Rings
        next_ring_idx = len(self.rings)
        while next_ring_idx < self.TOTAL_RINGS:
            ring_x = self.PLAYER_X_POS + self.RING_SPACING * (next_ring_idx + 1)
            if ring_x > end_x:
                break
            
            ring_y = self.np_random.integers(self.HEIGHT * 0.2, self.HEIGHT * 0.8)
            self.rings.append({
                "pos": [ring_x, ring_y],
                "collected": False,
                "radius": self.RING_RADIUS
            })
            next_ring_idx += 1
            
        # Generate Trees
        tree_density = self.INITIAL_TREE_DENSITY * (1.1 ** (self.rings_collected // 2))
        num_trees = int(chunk_width * tree_density)
        
        for _ in range(num_trees):
            tree_x = self.np_random.integers(start_x, end_x)
            too_close_to_ring = False
            for ring in self.rings:
                if abs(tree_x - ring["pos"][0]) < 100:
                    too_close_to_ring = True
                    break
            if too_close_to_ring:
                continue

            tree_height = self.np_random.integers(100, 300)
            on_top = self.np_random.choice([True, False])
            
            if on_top:
                tree_rect = pygame.Rect(tree_x, 0, 40, tree_height)
            else:
                tree_rect = pygame.Rect(tree_x, self.HEIGHT - tree_height, 40, tree_height)
            
            self.trees.append(tree_rect)
            
        self.world_generated_x = end_x
    
    def step(self, action):
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        # space_held = action[1] == 1  # Boolean
        # shift_held = action[2] == 1  # Boolean
        
        # --- Update game logic ---
        reward = 0.0
        
        # Player physics
        if movement == 1: # Up
            self.player_vy += self.LIFT
        elif movement == 2: # Down
            self.player_vy += self.DIVE
        
        self.player_vy += self.GRAVITY
        self.player_vy = np.clip(self.player_vy, -self.MAX_VEL, self.MAX_VEL)
        self.player_y += self.player_vy
        
        self.player_y = np.clip(self.player_y, self.PLAYER_SIZE, self.HEIGHT - self.PLAYER_SIZE)

        # World scrolling & particle updates
        for obj_list in [self.rings, self.particles]:
            for item in obj_list:
                item["pos"][0] -= self.SCROLL_SPEED
        for tree in self.trees:
            tree.x -= self.SCROLL_SPEED
        self._update_particles()
        
        # Generate bird trail
        if self.steps % 2 == 0:
            self.particles.append({
                "pos": [self.PLAYER_X_POS - self.PLAYER_SIZE, self.player_y],
                "vel": [-self.SCROLL_SPEED, self.np_random.uniform(-0.5, 0.5)],
                "life": 15,
                "radius": self.np_random.integers(2, 4),
                "color": (220, 220, 220)
            })

        # --- Collision detection and rewards ---
        player_rect = pygame.Rect(self.PLAYER_X_POS - self.PLAYER_SIZE, self.player_y - self.PLAYER_SIZE, self.PLAYER_SIZE * 2, self.PLAYER_SIZE * 2)
        
        collided_this_frame = False
        for tree in self.trees[:]:
            if player_rect.colliderect(tree):
                self.trees.remove(tree)
                self.trees_hit += 1
                reward = -5.0
                collided_this_frame = True
                self.screen_shake = 15
                # Sound: sfx_tree_hit.wav
                for _ in range(20):
                    self.particles.append({
                        "pos": list(player_rect.center),
                        "vel": [self.np_random.uniform(-3, 3) - self.SCROLL_SPEED, self.np_random.uniform(-3, 3)],
                        "life": 25, "radius": self.np_random.integers(2, 5), "color": self.COLOR_TREE_TRUNK
                    })
                break
        
        if not collided_this_frame:
            reward = 0.1 # Survival reward

        for ring in self.rings:
            if not ring["collected"]:
                dist = math.hypot(self.PLAYER_X_POS - ring["pos"][0], self.player_y - ring["pos"][1])
                if dist < ring["radius"]:
                    ring["collected"] = True
                    self.rings_collected += 1
                    reward = 10.0
                    # Sound: sfx_ring_collect.wav
                    for i in range(30):
                        angle = (i / 30) * 2 * math.pi
                        self.particles.append({
                            "pos": [ring["pos"][0], ring["pos"][1]],
                            "vel": [math.cos(angle) * 3 - self.SCROLL_SPEED, math.sin(angle) * 3],
                            "life": 30, "radius": self.np_random.integers(3, 6), "color": self.COLOR_RING
                        })

        # Cleanup & world generation
        self.trees = [tree for tree in self.trees if tree.right > 0]
        if self.world_generated_x - self.PLAYER_X_POS < self.WIDTH * 2:
            self._generate_world_chunk(self.WIDTH * 2)

        # --- Termination ---
        self.steps += 1
        terminated = False
        if self.rings_collected >= self.TOTAL_RINGS:
            terminated = True
            reward += 50.0 # Victory bonus
        elif self.trees_hit >= self.MAX_TREES_HIT:
            terminated = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True

        self.score += reward
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        # Apply screen shake
        render_offset = [0, 0]
        if self.screen_shake > 0:
            self.screen_shake -= 1
            render_offset[0] = self.np_random.integers(-5, 6)
            render_offset[1] = self.np_random.integers(-5, 6)

        # Background gradient
        for y in range(self.HEIGHT):
            interp = y / self.HEIGHT
            color = tuple(int(self.COLOR_SKY_TOP[i] * (1 - interp) + self.COLOR_SKY_BOTTOM[i] * interp) for i in range(3))
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))
        
        # Create a temporary surface for offset rendering
        offset_surface = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)

        # Rings
        for ring in self.rings:
            color = self.COLOR_RING if not ring["collected"] else self.COLOR_RING_COLLECTED
            pygame.gfxdraw.aacircle(offset_surface, int(ring["pos"][0]), int(ring["pos"][1]), ring["radius"], color)
            pygame.gfxdraw.aacircle(offset_surface, int(ring["pos"][0]), int(ring["pos"][1]), ring["radius"] - 3, color)

        # Trees
        for tree in self.trees:
            trunk = pygame.Rect(tree.x + tree.width * 0.25, tree.y, tree.width * 0.5, tree.height)
            leaves_h = tree.height * 0.7
            leaves_y = tree.y if tree.y == 0 else tree.bottom - leaves_h
            leaves = pygame.Rect(tree.x, leaves_y, tree.width, leaves_h)
            pygame.draw.rect(offset_surface, self.COLOR_TREE_TRUNK, trunk)
            pygame.draw.rect(offset_surface, self.COLOR_TREE_LEAVES, leaves)
        
        # Particles
        for p in self.particles:
            radius = max(0, int(p["radius"] * (p["life"] / 30)))
            if radius > 0:
                pygame.gfxdraw.filled_circle(offset_surface, int(p["pos"][0]), int(p["pos"][1]), radius, p["color"])
        
        # Player
        player_angle = math.atan2(self.player_vy, 15) * 0.5
        points = []
        for i in [0, 2.5, -2.5]:
            size_mod = 1.0 if i == 0 else 0.8
            angle = player_angle + i
            points.append((
                self.PLAYER_X_POS + math.cos(angle) * self.PLAYER_SIZE * size_mod,
                self.player_y + math.sin(angle) * self.PLAYER_SIZE * size_mod
            ))
        pygame.gfxdraw.aapolygon(offset_surface, points, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(offset_surface, points, self.COLOR_PLAYER)
        
        self.screen.blit(offset_surface, render_offset)

        # UI
        self._render_text(f"Rings: {self.rings_collected}/{self.TOTAL_RINGS}", (20, 20))
        self._render_text(f"Hits: {self.trees_hit}/{self.MAX_TREES_HIT}", (self.WIDTH - 120, 20))
        time_text = f"Steps: {self.steps}"
        time_width = self.font.size(time_text)[0]
        self._render_text(time_text, (self.WIDTH / 2 - time_width / 2, self.HEIGHT - 40))
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_text(self, text, pos):
        shadow = self.font.render(text, True, self.COLOR_UI_SHADOW)
        self.screen.blit(shadow, (pos[0] + 2, pos[1] + 2))
        message = self.font.render(text, True, self.COLOR_UI_TEXT)
        self.screen.blit(message, pos)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "rings_collected": self.rings_collected,
            "trees_hit": self.trees_hit,
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()

if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    terminated = False
    
    # Create a window to display the game
    pygame.display.set_caption("Forest Flight")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    action = np.array([0, 0, 0]) # no-op
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # --- Controls ---
        keys = pygame.key.get_pressed()
        movement = 0 # No-op
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        
        action = np.array([movement, 0, 0])
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated:
            print(f"Game Over! Final Info: {info}")
            obs, info = env.reset() # Restart the game
        
        # --- Display ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(env.FPS)
        
    env.close()