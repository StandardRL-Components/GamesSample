import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:03:44.272340
# Source Brief: brief_00187.md
# Brief Index: 187
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A stealth-survival game where the player teleports between quantum states
    to avoid enemy patrols. The player can create clones to act as diversions.
    The goal is to survive as long as possible.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "A stealth-survival game where the player teleports between quantum states to avoid enemy patrols. "
        "Create clones to act as diversions and survive as long as possible."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to teleport to adjacent unlocked nodes. "
        "Press space to create a clone as a diversion."
    )
    auto_advance = False

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    GRID_W, GRID_H = 16, 10
    CELL_SIZE = 40
    MAX_STEPS = 5000
    MAX_CLONES = 3
    CLONE_LIFESPAN = 10
    ENEMY_DETECTION_RADIUS = 1.5
    ENEMY_ATTRACTION_RADIUS = 5

    # Colors (Neon on Dark)
    COLOR_BG = (10, 15, 25)
    COLOR_GRID = (25, 35, 55)
    COLOR_PLAYER = (0, 192, 255)
    COLOR_PLAYER_GLOW = (0, 100, 200)
    COLOR_ENEMY = (255, 50, 50)
    COLOR_ENEMY_GLOW = (200, 0, 0)
    COLOR_CLONE = (100, 220, 255)
    COLOR_CLONE_GLOW = (80, 150, 200)
    COLOR_NODE_LOCKED = (60, 60, 80)
    COLOR_NODE_UNLOCKED = (50, 255, 150)
    COLOR_TEXT = (220, 220, 240)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 40, bold=True)

        # Game state variables are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = None
        self.enemies = []
        self.clones = []
        self.nodes = {}
        self.particles = []
        self.available_clones = 0
        self.base_enemy_speed = 0.5
        
        self.reset()
        
        # This check is critical as per the brief
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.particles = []
        self.available_clones = self.MAX_CLONES
        self.base_enemy_speed = 0.5

        self._setup_nodes()
        self._setup_initial_state()

        self.enemies = []
        self._spawn_enemy()
        
        self.clones = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0.1  # Survival reward for every step
        self.steps += 1

        # 1. Handle player action
        action_reward = self._handle_player_action(action)
        reward += action_reward
        
        # 2. Update game world state
        self._update_clones()
        detection, world_reward = self._update_enemies()
        reward += world_reward
        
        # 3. Check for game over from detection
        if detection:
            self.game_over = True
            reward = -100.0  # Detection penalty
            # sfx: player_detected_alarm
            self._create_particle_burst(self.player_pos, self.COLOR_ENEMY, 50)

        # 4. Handle game progression
        self._update_progression()

        # 5. Determine termination
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        if terminated and not self.game_over:
            reward = 100.0  # Survival victory reward
        
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_player_action(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        reward = 0
        
        original_pos = self.player_pos.copy()
        
        # Prioritize teleportation over cloning
        if movement != 0:
            target_pos = self.player_pos.copy()
            if movement == 1: target_pos.y -= 1 # Up
            elif movement == 2: target_pos.y += 1 # Down
            elif movement == 3: target_pos.x -= 1 # Left
            elif movement == 4: target_pos.x += 1 # Right

            tx, ty = int(target_pos.x), int(target_pos.y)
            if self.nodes.get((tx, ty), {}).get("unlocked", False):
                # Check for "dodge" reward
                is_near_enemy = any(original_pos.distance_to(e['pos']) < self.ENEMY_DETECTION_RADIUS + 1 for e in self.enemies)
                
                self.player_pos = target_pos
                
                if is_near_enemy:
                    reward += 1.0 # Successful dodge
                
                # sfx: player_teleport
                self._create_particle_burst(original_pos, self.COLOR_PLAYER, 20)
                self._create_particle_burst(self.player_pos, self.COLOR_PLAYER, 20, inward=True)
                
        if space_held and self.available_clones > 0:
            self.available_clones -= 1
            self.clones.append({
                "pos": self.player_pos.copy(),
                "lifespan": self.CLONE_LIFESPAN
            })
            # sfx: clone_created
            self._create_particle_burst(self.player_pos, self.COLOR_CLONE, 15)

        return reward

    def _update_clones(self):
        surviving_clones = []
        for clone in self.clones:
            clone["lifespan"] -= 1
            if clone["lifespan"] > 0:
                surviving_clones.append(clone)
            else:
                # sfx: clone_destroyed
                self._create_particle_burst(clone['pos'], self.COLOR_CLONE_GLOW, 10, lifespan=10)
        self.clones = surviving_clones

    def _update_enemies(self):
        detection = False
        reward = 0
        
        for enemy in self.enemies:
            # Check for player detection first
            if self.player_pos.distance_to(enemy['pos']) < self.ENEMY_DETECTION_RADIUS:
                detection = True
                # No need to move further if player is caught
                continue

            # Decide target: clone or path
            target_pos = None
            
            # Check for clone attraction
            closest_clone = None
            min_dist = float('inf')
            for clone in self.clones:
                dist = enemy['pos'].distance_to(clone['pos'])
                if dist < self.ENEMY_ATTRACTION_RADIUS and dist < min_dist:
                    min_dist = dist
                    closest_clone = clone
            
            if closest_clone:
                target_pos = closest_clone['pos']
            else:
                path_node_pos = pygame.Vector2(enemy['path'][enemy['path_index']])
                target_pos = path_node_pos
                if enemy['pos'].distance_to(path_node_pos) < 0.1:
                    enemy['path_index'] = (enemy['path_index'] + 1) % len(enemy['path'])
                    
            # Move towards target
            if target_pos != enemy['pos']:
                direction = (target_pos - enemy['pos']).normalize()
                enemy['pos'] += direction * enemy['speed']

            # Check for node unlocking
            ex, ey = round(enemy['pos'].x), round(enemy['pos'].y)
            if (ex, ey) in enemy['path'] and not self.nodes.get((ex, ey), {}).get("unlocked", True):
                self.nodes[(ex, ey)]["unlocked"] = True
                reward += 5.0
                # sfx: node_unlocked
                self._create_particle_burst(pygame.Vector2(ex, ey), self.COLOR_NODE_UNLOCKED, 25, lifespan=15)

        return detection, reward

    def _update_progression(self):
        if self.steps > 0:
            # Increase enemy speed
            if self.steps % 200 == 0:
                self.base_enemy_speed = min(1.0, self.base_enemy_speed + 0.05)
                for e in self.enemies:
                    e['speed'] = self.base_enemy_speed
            # Spawn new enemy
            if self.steps % 500 == 0:
                self._spawn_enemy()
    
    def _setup_nodes(self):
        self.nodes = {}
        for y in range(self.GRID_H):
            for x in range(self.GRID_W):
                self.nodes[(x, y)] = {"unlocked": False}

    def _setup_initial_state(self):
        self.player_pos = pygame.Vector2(self.GRID_W // 2, self.GRID_H // 2)
        # Unlock a 3x3 area around the player
        for y in range(int(self.player_pos.y) - 1, int(self.player_pos.y) + 2):
            for x in range(int(self.player_pos.x) - 1, int(self.player_pos.x) + 2):
                if 0 <= x < self.GRID_W and 0 <= y < self.GRID_H:
                    self.nodes[(x, y)]["unlocked"] = True

    def _spawn_enemy(self):
        # Spawn away from player
        spawn_edge = self.np_random.choice(['top', 'bottom', 'left', 'right'])
        if spawn_edge == 'top': start_pos = pygame.Vector2(self.np_random.integers(0, self.GRID_W), 0)
        elif spawn_edge == 'bottom': start_pos = pygame.Vector2(self.np_random.integers(0, self.GRID_W), self.GRID_H - 1)
        elif spawn_edge == 'left': start_pos = pygame.Vector2(0, self.np_random.integers(0, self.GRID_H))
        else: start_pos = pygame.Vector2(self.GRID_W - 1, self.np_random.integers(0, self.GRID_H))

        # Create a simple patrol path
        path = [start_pos.copy()]
        current_pos = start_pos.copy()
        for _ in range(self.np_random.integers(3, 7)):
            next_pos = pygame.Vector2(
                self.np_random.integers(0, self.GRID_W),
                self.np_random.integers(0, self.GRID_H)
            )
            path.append(next_pos)
        
        self.enemies.append({
            "pos": start_pos,
            "path": path,
            "path_index": 0,
            "speed": self.base_enemy_speed
        })
        # sfx: enemy_spawned
    
    def _grid_to_pixel(self, pos):
        x = int((pos.x + 0.5) * self.CELL_SIZE)
        y = int((pos.y + 0.5) * self.CELL_SIZE)
        return x, y

    def _draw_glow(self, surface, color, center, radius, steps=5):
        for i in range(steps, 0, -1):
            alpha = int(100 * (i / steps)**2)
            temp_radius = int(radius * (1 + (steps - i) * 0.15))
            pygame.gfxdraw.filled_circle(surface, center[0], center[1], temp_radius, (*color, alpha))
    
    def _create_particle_burst(self, grid_pos, color, count, lifespan=20, inward=False):
        pixel_pos = self._grid_to_pixel(grid_pos)
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            if inward: vel *= -1
            self.particles.append({
                "pos": pygame.Vector2(pixel_pos),
                "vel": vel,
                "lifespan": self.np_random.integers(lifespan//2, lifespan),
                "max_lifespan": lifespan,
                "color": color,
                "size": self.np_random.uniform(1, 4)
            })

    def _update_and_render_particles(self):
        active_particles = []
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            if p['lifespan'] > 0:
                active_particles.append(p)
                alpha = int(255 * (p['lifespan'] / p['max_lifespan']))
                size = int(p['size'] * (p['lifespan'] / p['max_lifespan']))
                if size > 0:
                    surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                    pygame.draw.circle(surf, (*p['color'], alpha), (size, size), size)
                    self.screen.blit(surf, (p['pos'].x - size, p['pos'].y - size), special_flags=pygame.BLEND_RGBA_ADD)
        self.particles = active_particles

    def _render_game(self):
        # Background grid
        for x in range(0, self.WIDTH, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

        # Teleport Nodes
        for (x, y), data in self.nodes.items():
            px, py = self._grid_to_pixel(pygame.Vector2(x, y))
            color = self.COLOR_NODE_UNLOCKED if data["unlocked"] else self.COLOR_NODE_LOCKED
            pygame.gfxdraw.filled_circle(self.screen, px, py, 4, color)
            pygame.gfxdraw.aacircle(self.screen, px, py, 4, color)

        # Update and render particles
        self._update_and_render_particles()

        # Clones
        for clone in self.clones:
            pos_px = self._grid_to_pixel(clone['pos'])
            radius = int(self.CELL_SIZE * 0.3)
            alpha_mult = (clone['lifespan'] / self.CLONE_LIFESPAN)
            self._draw_glow(self.screen, self.COLOR_CLONE_GLOW, pos_px, int(radius*1.5), steps=3)
            pygame.gfxdraw.filled_circle(self.screen, pos_px[0], pos_px[1], radius, (*self.COLOR_CLONE, int(255*alpha_mult)))
            pygame.gfxdraw.aacircle(self.screen, pos_px[0], pos_px[1], radius, (*self.COLOR_CLONE, int(255*alpha_mult)))

        # Enemies
        for enemy in self.enemies:
            pos_px = self._grid_to_pixel(enemy['pos'])
            radius = int(self.CELL_SIZE * 0.35)
            self._draw_glow(self.screen, self.COLOR_ENEMY_GLOW, pos_px, int(radius*1.8))
            
            # Draw diamond shape
            points = [
                (pos_px[0], pos_px[1] - radius),
                (pos_px[0] + radius, pos_px[1]),
                (pos_px[0], pos_px[1] + radius),
                (pos_px[0] - radius, pos_px[1]),
            ]
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_ENEMY)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ENEMY)

        # Player
        if not self.game_over:
            pos_px = self._grid_to_pixel(self.player_pos)
            radius = int(self.CELL_SIZE * 0.4)
            self._draw_glow(self.screen, self.COLOR_PLAYER_GLOW, pos_px, int(radius * 2))
            pygame.gfxdraw.filled_circle(self.screen, pos_px[0], pos_px[1], radius, self.COLOR_PLAYER)
            pygame.gfxdraw.aacircle(self.screen, pos_px[0], pos_px[1], radius, self.COLOR_PLAYER)

    def _render_ui(self):
        # Steps/Time display
        steps_text = self.font_ui.render(f"TIME: {self.steps}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (10, 10))

        # Clones display
        clone_text = self.font_ui.render(f"CLONES: {self.available_clones}", True, self.COLOR_TEXT)
        self.screen.blit(clone_text, (self.WIDTH - clone_text.get_width() - 10, 10))
        
        # Game Over message
        if self.game_over:
            s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            s.fill((0,0,0,180))
            self.screen.blit(s, (0,0))
            
            status = "SURVIVED" if self.steps >= self.MAX_STEPS else "DETECTED"
            end_text = self.font_game_over.render(status, True, self.COLOR_TEXT)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

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
            "player_pos": (self.player_pos.x, self.player_pos.y),
            "enemies": len(self.enemies),
            "clones": len(self.clones)
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc is False
        assert isinstance(info, dict)
        
        # print("✓ Implementation validated successfully")

# Example usage to test the environment visually
if __name__ == "__main__":
    # Un-comment the line below to run with a display
    # os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv()
    obs, info = env.reset()
    
    # Setup for manual play
    try:
        pygame.display.init()
        pygame.font.init()
        screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
        pygame.display.set_caption("Quantum Leap")
        clock = pygame.time.Clock()
        display_active = True
    except pygame.error:
        print("No display available. Running headlessly.")
        display_active = False

    done = False
    total_reward = 0
    
    print("\n--- Manual Control ---")
    print("Arrows: Teleport")
    print("Space: Create Clone")
    print("Q: Quit")
    
    while not done:
        # Default action is "do nothing"
        movement = 0
        space = 0
        shift = 0
        
        if display_active:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                    done = True

            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            if keys[pygame.K_SPACE]: space = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
            action = [movement, space, shift]
        else: # If no display, just sample actions
            action = env.action_space.sample()
            if env.steps > 200: # Limit headless run
                done = True

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

        if display_active:
            # Render the observation to the display
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            clock.tick(10) # Control game speed for manual play
        
    print(f"Game Over! Final Score: {total_reward:.2f}, Steps: {info['steps']}")
    
    env.close()