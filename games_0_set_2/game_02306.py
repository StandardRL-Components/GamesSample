
# Generated: 2025-08-27T19:57:52.578581
# Source Brief: brief_02306.md
# Brief Index: 2306

        
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

    user_guide = (
        "Controls: Arrow keys to move on the isometric grid. Dodge the purple enemies and collect the sparkling gems."
    )

    game_description = (
        "Collect sparkling gems while dodging cunning enemies in a vibrant isometric world. Collect 20 gems to win, but lose all your health and it's game over. Get bonus points for risky grabs near enemies!"
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.screen_width = 640
        self.screen_height = 400
        self.world_size = (40, 40) # Isometric grid dimensions
        self.max_steps = 1000
        self.win_gems = 20
        self.max_health = 3
        self.initial_enemies = 3
        self.max_enemies = 6
        self.initial_gems = 5
        self.enemy_chase_radius = 8
        self.enemy_collision_invuln_steps = 60 # 2 seconds at 30fps
        self.risky_gem_radius = 5

        # Visual constants
        self.tile_width_iso = 48
        self.tile_height_iso = 24
        
        # Colors
        self.COLOR_BG = (18, 22, 33)
        self.COLOR_GRID = (35, 45, 63)
        self.COLOR_PLAYER = (255, 215, 0)
        self.COLOR_PLAYER_SHADOW = (0, 0, 0, 100)
        self.COLOR_ENEMY = (138, 43, 226)
        self.COLOR_ENEMY_CHASE = (220, 20, 60)
        self.COLOR_GEM_RED = (255, 50, 50)
        self.COLOR_GEM_GREEN = (50, 255, 50)
        self.COLOR_GEM_BLUE = (50, 150, 255)
        self.COLOR_UI_TEXT = (230, 230, 230)
        self.COLOR_HEART = (255, 80, 80)
        
        # Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.screen_height, self.screen_width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 32)
        self.font_game_over = pygame.font.Font(None, 72)
        
        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player = {}
        self.enemies = []
        self.gems = []
        self.particles = []
        self.gems_collected = 0
        self.enemy_speed = 0.0
        self.player_invuln_timer = 0
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.gems_collected = 0
        self.game_over = False
        self.player_invuln_timer = 0
        self.enemy_speed = 1.0 / 30.0 # 1 tile per 30 frames (1 second)

        self.player = {
            "iso_pos": np.array([self.world_size[0] / 2, self.world_size[1] / 2], dtype=float),
            "health": self.max_health,
            "radius": 0.4
        }
        
        self.enemies = []
        for _ in range(self.initial_enemies):
            self._spawn_enemy()

        self.gems = []
        for _ in range(self.initial_gems):
            self._spawn_gem()
            
        self.particles = []
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        terminated = False
        
        # Find closest entities before action for reward calculation
        closest_gem_before, dist_gem_before = self._get_closest_entity(self.gems)
        closest_enemy_before, dist_enemy_before = self._get_closest_entity(self.enemies)

        if not self.game_over:
            movement = action[0]
            self._handle_player_movement(movement)
            self._update_enemies()
        
        self._update_particles()
        
        if not self.game_over:
            # Movement-based reward
            closest_gem_after, dist_gem_after = self._get_closest_entity(self.gems)
            closest_enemy_after, dist_enemy_after = self._get_closest_entity(self.enemies)

            if dist_gem_before is not None and dist_gem_after is not None:
                reward += (dist_gem_before - dist_gem_after) * 0.1
            if dist_enemy_before is not None and dist_enemy_after is not None:
                # We want to reward moving away from enemies
                reward += (dist_enemy_after - dist_enemy_before) * 0.2

            # Collision detection and event-based rewards
            collision_reward = self._check_collisions()
            reward += collision_reward

        # Update timers
        if self.player_invuln_timer > 0:
            self.player_invuln_timer -= 1
        
        self.steps += 1
        
        # Check termination conditions
        if self.gems_collected >= self.win_gems:
            terminated = True
            reward += 100
            self.game_over = "WIN"
        elif self.player["health"] <= 0:
            terminated = True
            reward -= 100
            self.game_over = "LOSE"
        elif self.steps >= self.max_steps:
            terminated = True
            self.game_over = "TIMEOUT"
            
        return (
            self._get_observation(),
            np.clip(reward, -100, 100),
            terminated,
            False,
            self._get_info()
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        # Center camera on player
        player_cart_pos = self._iso_to_cart(self.player["iso_pos"])
        camera_offset = player_cart_pos - np.array([self.screen_width / 2, self.screen_height / 2])
        
        self._render_game(camera_offset)
        self._render_ui()
        
        if self.game_over:
            self._render_game_over()
            
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.player["health"],
            "gems_collected": self.gems_collected,
        }

    # --- Game Logic Helpers ---

    def _handle_player_movement(self, movement):
        move_vec = np.array([0.0, 0.0])
        speed = 1.0 / 6.0 # Move one tile in 6 frames
        if movement == 1: move_vec = np.array([0, -1]) # Up (NW)
        elif movement == 2: move_vec = np.array([0, 1])  # Down (SE)
        elif movement == 3: move_vec = np.array([-1, 0]) # Left (SW)
        elif movement == 4: move_vec = np.array([1, 0])  # Right (NE)
        
        self.player["iso_pos"] += move_vec * speed
        
        # Clamp to world boundaries
        self.player["iso_pos"][0] = np.clip(self.player["iso_pos"][0], 0, self.world_size[0])
        self.player["iso_pos"][1] = np.clip(self.player["iso_pos"][1], 0, self.world_size[1])

    def _update_enemies(self):
        player_pos = self.player["iso_pos"]
        for enemy in self.enemies:
            dist_to_player = np.linalg.norm(player_pos - enemy["iso_pos"])
            
            # State transition: PATROL <-> CHASE
            if dist_to_player < self.enemy_chase_radius and enemy["state"] == "PATROL":
                enemy["state"] = "CHASE"
            elif dist_to_player >= self.enemy_chase_radius * 1.2 and enemy["state"] == "CHASE":
                enemy["state"] = "PATROL"
                enemy["target_pos"] = self._get_random_world_pos() # New patrol point
            
            # Movement
            if enemy["state"] == "CHASE":
                enemy["target_pos"] = player_pos

            if np.linalg.norm(enemy["iso_pos"] - enemy["target_pos"]) < 0.5:
                if enemy["state"] == "PATROL":
                    enemy["target_pos"] = self._get_random_world_pos()
            else:
                direction = (enemy["target_pos"] - enemy["iso_pos"])
                norm = np.linalg.norm(direction)
                if norm > 0:
                    enemy["iso_pos"] += (direction / norm) * self.enemy_speed

    def _update_particles(self):
        self.particles = [p for p in self.particles if p["life"] > 0]
        for p in self.particles:
            p["pos"] += p["vel"]
            p["life"] -= 1
            p["radius"] = max(0, p["radius"] - 0.1)

    def _check_collisions(self):
        reward = 0
        
        # Player-Enemy
        if self.player_invuln_timer == 0:
            for enemy in self.enemies:
                dist = np.linalg.norm(self.player["iso_pos"] - enemy["iso_pos"])
                if dist < self.player["radius"] + enemy["radius"]:
                    # SFX: Player Hit
                    self.player["health"] -= 1
                    reward -= 5
                    self.player_invuln_timer = self.enemy_collision_invuln_steps
                    self._create_particles(self.player["iso_pos"], 20, self.COLOR_HEART)
                    break # Only one hit per frame
        
        # Player-Gem
        gems_to_remove = []
        for gem in self.gems:
            dist = np.linalg.norm(self.player["iso_pos"] - gem["iso_pos"])
            if dist < self.player["radius"] + gem["radius"]:
                # SFX: Gem Collect
                gems_to_remove.append(gem)
                self.gems_collected += 1
                self.score += 1
                reward += 1
                
                # Risky grab bonus
                _, dist_to_enemy = self._get_closest_entity(self.enemies, self.player["iso_pos"])
                if dist_to_enemy is not None and dist_to_enemy < self.risky_gem_radius:
                    reward += 2
                    self.score += 2
                    # Create special particles for risky grab
                    self._create_particles(gem["iso_pos"], 30, self.COLOR_PLAYER)
                else:
                    self._create_particles(gem["iso_pos"], 15, gem["color"])
        
        if gems_to_remove:
            self.gems = [g for g in self.gems if g not in gems_to_remove]
            for _ in range(len(gems_to_remove)):
                self._spawn_gem()

            # Difficulty scaling
            if self.gems_collected > 0 and self.gems_collected % 5 == 0:
                self.enemy_speed += 0.05
                if len(self.enemies) < self.max_enemies:
                    self._spawn_enemy()
        
        return reward

    def _spawn_gem(self):
        gem_type = self.np_random.choice(["red", "green", "blue"])
        colors = {"red": self.COLOR_GEM_RED, "green": self.COLOR_GEM_GREEN, "blue": self.COLOR_GEM_BLUE}
        self.gems.append({
            "iso_pos": self._get_random_world_pos(min_dist_from_player=5),
            "radius": 0.3,
            "color": colors[gem_type],
            "type": "gem"
        })

    def _spawn_enemy(self):
        self.enemies.append({
            "iso_pos": self._get_random_world_pos(min_dist_from_player=8),
            "radius": 0.45,
            "state": "PATROL",
            "target_pos": self._get_random_world_pos(),
            "type": "enemy"
        })

    def _get_random_world_pos(self, min_dist_from_player=0.0):
        while True:
            pos = self.np_random.random(2) * self.world_size
            if np.linalg.norm(pos - self.player["iso_pos"]) >= min_dist_from_player:
                return pos

    def _get_closest_entity(self, entities, from_pos=None):
        if not entities:
            return None, None
        
        pos = from_pos if from_pos is not None else self.player["iso_pos"]
        
        closest_entity = None
        min_dist = float('inf')
        
        for entity in entities:
            dist = np.linalg.norm(entity["iso_pos"] - pos)
            if dist < min_dist:
                min_dist = dist
                closest_entity = entity
        return closest_entity, min_dist

    def _create_particles(self, iso_pos, count, color):
        cart_pos = self._iso_to_cart(iso_pos)
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            self.particles.append({
                "pos": cart_pos.copy(),
                "vel": vel,
                "life": self.np_random.integers(15, 30),
                "radius": self.np_random.uniform(2, 5),
                "color": color
            })

    # --- Rendering Helpers ---

    def _iso_to_cart(self, iso_pos):
        x = (iso_pos[0] - iso_pos[1]) * self.tile_width_iso / 2
        y = (iso_pos[0] + iso_pos[1]) * self.tile_height_iso / 2
        return np.array([x, y])
    
    def _render_game(self, camera_offset):
        # Grid
        for i in range(self.world_size[0] + 1):
            start_iso = np.array([i, 0.0])
            end_iso = np.array([i, float(self.world_size[1])])
            start_cart = self._iso_to_cart(start_iso) - camera_offset
            end_cart = self._iso_to_cart(end_iso) - camera_offset
            pygame.draw.aaline(self.screen, self.COLOR_GRID, start_cart, end_cart)

        for i in range(self.world_size[1] + 1):
            start_iso = np.array([0.0, i])
            end_iso = np.array([float(self.world_size[0]), i])
            start_cart = self._iso_to_cart(start_iso) - camera_offset
            end_cart = self._iso_to_cart(end_iso) - camera_offset
            pygame.draw.aaline(self.screen, self.COLOR_GRID, start_cart, end_cart)

        # Entities
        entities = self.enemies + self.gems + [self.player]
        # Sort by Y-axis for correct painter's algorithm rendering
        entities.sort(key=lambda e: e["iso_pos"][0] + e["iso_pos"][1])

        # Shadows
        for entity in entities:
            cart_pos = self._iso_to_cart(entity["iso_pos"])
            screen_pos = cart_pos - camera_offset
            shadow_radius = int(entity["radius"] * self.tile_width_iso / 2 * 1.1)
            shadow_rect = pygame.Rect(0, 0, shadow_radius * 2, shadow_radius)
            shadow_rect.center = (int(screen_pos[0]), int(screen_pos[1] + self.tile_height_iso * 0.2))
            shadow_surf = pygame.Surface(shadow_rect.size, pygame.SRCALPHA)
            pygame.draw.ellipse(shadow_surf, self.COLOR_PLAYER_SHADOW, (0, 0, shadow_rect.width, shadow_rect.height))
            self.screen.blit(shadow_surf, shadow_rect.topleft)

        # Main bodies
        for entity in entities:
            bob_offset = math.sin(self.steps * 0.1 + (entity["iso_pos"][0] % 5)) * 4
            cart_pos = self._iso_to_cart(entity["iso_pos"])
            screen_pos = cart_pos - camera_offset
            screen_pos[1] -= bob_offset
            
            radius = int(entity["radius"] * self.tile_width_iso / 2)
            
            if entity.get("type") == "gem":
                pulse = (math.sin(self.steps * 0.2 + entity["iso_pos"][1]) + 1) / 2
                glow_radius = int(radius * (1.2 + pulse * 0.5))
                pygame.gfxdraw.filled_circle(self.screen, int(screen_pos[0]), int(screen_pos[1]), glow_radius, (*entity["color"], 60))
                pygame.gfxdraw.filled_circle(self.screen, int(screen_pos[0]), int(screen_pos[1]), radius, entity["color"])
                pygame.gfxdraw.aacircle(self.screen, int(screen_pos[0]), int(screen_pos[1]), radius, (255,255,255))
            elif entity.get("type") == "enemy":
                color = self.COLOR_ENEMY_CHASE if entity["state"] == "CHASE" else self.COLOR_ENEMY
                pygame.gfxdraw.filled_circle(self.screen, int(screen_pos[0]), int(screen_pos[1]), radius, color)
                pygame.gfxdraw.aacircle(self.screen, int(screen_pos[0]), int(screen_pos[1]), radius, tuple(min(255, c+50) for c in color))
            else: # Player
                color = self.COLOR_PLAYER
                if self.player_invuln_timer > 0 and self.steps % 4 < 2:
                    color = (min(255, c + 100) for c in self.COLOR_PLAYER)
                pygame.gfxdraw.filled_circle(self.screen, int(screen_pos[0]), int(screen_pos[1]), radius, color)
                pygame.gfxdraw.aacircle(self.screen, int(screen_pos[0]), int(screen_pos[1]), radius, (255,255,255))

        # Particles
        for p in self.particles:
            screen_pos = p["pos"] - camera_offset
            if p["radius"] > 0:
                alpha = int(255 * (p["life"] / 30.0))
                color = (*p["color"], alpha)
                pygame.gfxdraw.filled_circle(self.screen, int(screen_pos[0]), int(screen_pos[1]), int(p["radius"]), color)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Gems collected
        gem_text = self.font_ui.render(f"GEMS: {self.gems_collected}/{self.win_gems}", True, self.COLOR_UI_TEXT)
        self.screen.blit(gem_text, (10, 40))

        # Health
        for i in range(self.max_health):
            heart_pos = (self.screen_width - 35 * (i + 1), 15)
            if i < self.player["health"]:
                color = self.COLOR_HEART
            else:
                color = self.COLOR_GRID
            pygame.gfxdraw.filled_polygon(self.screen, [
                (heart_pos[0] + 15, heart_pos[1] + 10), (heart_pos[0], heart_pos[1] + 25), (heart_pos[0] + 30, heart_pos[1] + 25)
            ], color)
            pygame.gfxdraw.filled_circle(self.screen, heart_pos[0] + 8, heart_pos[1] + 10, 10, color)
            pygame.gfxdraw.filled_circle(self.screen, heart_pos[0] + 22, heart_pos[1] + 10, 10, color)

    def _render_game_over(self):
        overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        
        if self.game_over == "WIN":
            text = "YOU WIN!"
            color = self.COLOR_PLAYER
        else:
            text = "GAME OVER"
            color = self.COLOR_HEART
            
        text_surf = self.font_game_over.render(text, True, color)
        text_rect = text_surf.get_rect(center=(self.screen_width / 2, self.screen_height / 2))
        self.screen.blit(overlay, (0, 0))
        self.screen.blit(text_surf, text_rect)

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.screen_height, self.screen_width, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.screen_height, self.screen_width, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.screen_height, self.screen_width, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Gem Collector")
    screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    clock = pygame.time.Clock()
    
    terminated = False
    
    print(env.user_guide)
    
    while not terminated:
        # Get human input
        movement = 0 # no-op
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
                
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
            
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [movement, space_held, shift_held]
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Cap the frame rate
        clock.tick(30)
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}")
            # Wait a bit before closing
            pygame.time.wait(3000)

    env.close()