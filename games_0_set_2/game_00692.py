import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move your character. "
        "Collect all the yellow gems while avoiding the red enemies."
    )

    game_description = (
        "An isometric arcade game where you must collect gems and dodge enemies. "
        "Reach the target score to win, but watch your health!"
    )

    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 20, 15
    TILE_WIDTH, TILE_HEIGHT = 32, 16
    
    # Colors
    COLOR_BG = (20, 30, 40)
    COLOR_GRID = (40, 50, 60)
    COLOR_PLAYER = (0, 255, 150)
    COLOR_PLAYER_GLOW = (0, 255, 150, 50)
    COLOR_ENEMY = (255, 50, 50)
    COLOR_ENEMY_GLOW = (255, 50, 50, 70)
    COLOR_GEM = (255, 220, 0)
    COLOR_GEM_GLOW = (255, 220, 0, 100)
    COLOR_TEXT = (220, 220, 240)
    
    # Game parameters
    INITIAL_HEALTH = 3
    GEMS_TO_WIN = 20
    MAX_STEPS = 1500
    NUM_ENEMIES = 5
    ENEMY_CHASE_RADIUS = 5
    ENEMY_MOVE_INTERVAL = 3  # Move every N steps
    PLAYER_MOVE_INTERVAL = 2 # Move every N steps
    INVULNERABILITY_FRAMES = 30 # After getting hit

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)
        
        # This offset centers the grid on the screen
        self.grid_offset_x = self.SCREEN_WIDTH / 2
        self.grid_offset_y = self.SCREEN_HEIGHT / 2 - (self.GRID_HEIGHT * self.TILE_HEIGHT / 2) + 50

        # All state variables are initialized in reset()
        self.player = {}
        self.enemies = []
        self.gems = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.health = 0
        self.game_over = False
        self.player_move_cooldown = 0
        self.invulnerability_timer = 0
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.health = self.INITIAL_HEALTH
        self.game_over = False
        self.particles = []
        self.player_move_cooldown = 0
        self.invulnerability_timer = 0

        # Player setup
        start_pos = (self.GRID_WIDTH // 2, self.GRID_HEIGHT - 2)
        visual_pos = self._cart_to_iso(*start_pos)
        self.player = {
            "pos": np.array(start_pos),
            "visual_pos": np.array(visual_pos, dtype=float),
            "target_visual_pos": np.array(visual_pos, dtype=float)
        }

        # Procedural generation of gems and enemies
        self.gems = []
        occupied_coords = {tuple(self.player["pos"])}
        
        while len(self.gems) < self.GEMS_TO_WIN:
            pos = (self.np_random.integers(0, self.GRID_WIDTH), self.np_random.integers(0, self.GRID_HEIGHT))
            if pos not in occupied_coords:
                visual_pos = self._cart_to_iso(*pos)
                self.gems.append({
                    "pos": np.array(pos),
                    "visual_pos": np.array(visual_pos, dtype=float),
                    "pulse": self.np_random.random() * math.pi * 2
                })
                occupied_coords.add(pos)

        self.enemies = []
        for i in range(self.NUM_ENEMIES):
            while True:
                pos = (self.np_random.integers(0, self.GRID_WIDTH), self.np_random.integers(0, self.GRID_HEIGHT // 2))
                if pos not in occupied_coords:
                    visual_pos = self._cart_to_iso(*pos)
                    enemy_type = self.np_random.choice(['patrol', 'chase', 'random'])
                    
                    enemy = {
                        "pos": np.array(pos),
                        "visual_pos": np.array(visual_pos, dtype=float),
                        "target_visual_pos": np.array(visual_pos, dtype=float),
                        "type": enemy_type,
                        "move_cooldown": self.np_random.integers(0, self.ENEMY_MOVE_INTERVAL),
                    }

                    if enemy_type == 'patrol':
                        p1 = pos
                        p2 = (pos[0] + self.np_random.integers(-4, 5), pos[1] + self.np_random.integers(-3, 4))
                        p2 = (np.clip(p2[0], 0, self.GRID_WIDTH - 1), np.clip(p2[1], 0, self.GRID_HEIGHT - 1))
                        enemy['path'] = [np.array(p1), np.array(p2)]
                        enemy['path_target'] = 1

                    self.enemies.append(enemy)
                    occupied_coords.add(pos)
                    break

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        movement = action[0]
        
        self.steps += 1
        if self.player_move_cooldown > 0: self.player_move_cooldown -= 1
        if self.invulnerability_timer > 0: self.invulnerability_timer -= 1
        
        # --- Distance-to-gem reward calculation (Part 1) ---
        dist_before = self._get_distance_to_nearest_gem()

        # --- Player Movement ---
        if movement != 0 and self.player_move_cooldown == 0:
            new_pos = self.player["pos"].copy()
            if movement == 1: new_pos[1] -= 1  # Up
            elif movement == 2: new_pos[1] += 1  # Down
            elif movement == 3: new_pos[0] -= 1  # Left
            elif movement == 4: new_pos[0] += 1  # Right
            
            # Clamp to grid boundaries
            new_pos[0] = np.clip(new_pos[0], 0, self.GRID_WIDTH - 1)
            new_pos[1] = np.clip(new_pos[1], 0, self.GRID_HEIGHT - 1)
            
            self.player["pos"] = new_pos
            self.player["target_visual_pos"] = self._cart_to_iso(*self.player["pos"])
            self.player_move_cooldown = self.PLAYER_MOVE_INTERVAL

        # --- Distance-to-gem reward calculation (Part 2) ---
        dist_after = self._get_distance_to_nearest_gem()
        if dist_after < dist_before:
            reward += 0.1
        elif dist_after > dist_before:
            reward -= 0.1
        
        # --- Enemy AI and Movement ---
        self._update_enemies()

        # --- Collision Detection ---
        # Player vs Gems
        gem_to_remove_index = -1
        for i, gem in enumerate(self.gems):
            if np.array_equal(self.player["pos"], gem["pos"]):
                gem_to_remove_index = i
                break
        
        if gem_to_remove_index != -1:
            # FIX: Use pop(index) instead of remove(object).
            # list.remove(obj) fails because comparing dictionaries containing numpy arrays
            # leads to a "ValueError: The truth value of an array with more than one element is ambiguous".
            # By finding the index and using pop, we avoid this problematic comparison.
            self.gems.pop(gem_to_remove_index)
            self.score += 1
            reward += 1.0
            self._create_particles(self.player["visual_pos"], self.COLOR_GEM, 20)
            # Placeholder: # Play gem collect sound

        # Player vs Enemies
        if self.invulnerability_timer == 0:
            for enemy in self.enemies:
                if np.array_equal(self.player["pos"], enemy["pos"]):
                    self.health -= 1
                    reward -= 5.0
                    self.invulnerability_timer = self.INVULNERABILITY_FRAMES
                    self._create_particles(self.player["visual_pos"], self.COLOR_ENEMY, 30, 2.5)
                    # Placeholder: # Play damage sound
                    break
        
        # --- Update visual positions (interpolation) ---
        self._interpolate_visuals()
        
        # --- Update particles ---
        self._update_particles()
        
        # --- Termination Conditions ---
        terminated = False
        truncated = False
        if self.health <= 0:
            reward -= 100.0
            terminated = True
            self.game_over = True
        if self.score >= self.GEMS_TO_WIN:
            reward += 100.0
            terminated = True
            self.game_over = True
        if self.steps >= self.MAX_STEPS:
            truncated = True # Use truncated for time limit
            self.game_over = True
            
        return (
            self._get_observation(),
            np.clip(reward, -100.0, 100.0), # Ensure reward scale
            terminated,
            truncated,
            self._get_info()
        )

    # --- Helper and Rendering Methods ---

    def _cart_to_iso(self, x, y):
        iso_x = (x - y) * (self.TILE_WIDTH / 2) + self.grid_offset_x
        iso_y = (x + y) * (self.TILE_HEIGHT / 2) + self.grid_offset_y
        return iso_x, iso_y

    def _draw_iso_diamond(self, surface, color, x, y, glow_color=None):
        points = [
            (x, y - self.TILE_HEIGHT / 2),
            (x + self.TILE_WIDTH / 2, y),
            (x, y + self.TILE_HEIGHT / 2),
            (x - self.TILE_WIDTH / 2, y)
        ]
        pygame.draw.polygon(surface, color, points)
        pygame.gfxdraw.aapolygon(surface, points, color)
        
        if glow_color:
            pygame.gfxdraw.filled_circle(surface, int(x), int(y), int(self.TILE_WIDTH/2 * 1.2), glow_color)


    def _get_distance_to_nearest_gem(self):
        if not self.gems:
            return 0
        player_pos = self.player["pos"]
        distances = [np.linalg.norm(player_pos - gem["pos"]) for gem in self.gems]
        return min(distances)

    def _update_enemies(self):
        for enemy in self.enemies:
            enemy["move_cooldown"] -= 1
            if enemy["move_cooldown"] > 0:
                continue
            
            enemy["move_cooldown"] = self.ENEMY_MOVE_INTERVAL
            current_pos = enemy["pos"]
            target_pos = current_pos.copy()

            if enemy['type'] == 'chase':
                dist_to_player = np.linalg.norm(current_pos - self.player['pos'])
                if dist_to_player < self.ENEMY_CHASE_RADIUS:
                    # Move towards player
                    delta = self.player['pos'] - current_pos
                    if abs(delta[0]) > abs(delta[1]):
                        target_pos[0] += np.sign(delta[0])
                    else:
                        target_pos[1] += np.sign(delta[1])
                else: # Wander randomly when not chasing
                    axis = self.np_random.choice([0, 1])
                    direction = self.np_random.choice([-1, 1])
                    target_pos[axis] += direction

            elif enemy['type'] == 'patrol':
                path_target_pos = enemy['path'][enemy['path_target']]
                if np.array_equal(current_pos, path_target_pos):
                    enemy['path_target'] = 1 - enemy['path_target'] # Flip target
                
                delta = enemy['path'][enemy['path_target']] - current_pos
                if np.any(delta): # Check if not already at target
                    if abs(delta[0]) > abs(delta[1]):
                        target_pos[0] += np.sign(delta[0])
                    else:
                        target_pos[1] += np.sign(delta[1])

            elif enemy['type'] == 'random':
                axis = self.np_random.choice([0, 1])
                direction = self.np_random.choice([-1, 1])
                target_pos[axis] += direction

            # Clamp and update
            target_pos[0] = np.clip(target_pos[0], 0, self.GRID_WIDTH - 1)
            target_pos[1] = np.clip(target_pos[1], 0, self.GRID_HEIGHT - 1)
            enemy['pos'] = target_pos
            enemy['target_visual_pos'] = self._cart_to_iso(*target_pos)

    def _interpolate_visuals(self):
        lerp_rate = 0.4 # Higher is faster/snappier
        self.player["visual_pos"] += (self.player["target_visual_pos"] - self.player["visual_pos"]) * lerp_rate
        for entity in self.enemies: # Gems are handled separately for pulsing
            if "target_visual_pos" in entity:
                 entity["visual_pos"] += (entity["target_visual_pos"] - entity["visual_pos"]) * lerp_rate
        for gem in self.gems:
            if "target_visual_pos" in gem:
                 gem["visual_pos"] += (gem["target_visual_pos"] - gem["visual_pos"]) * lerp_rate
            if "pulse" in gem:
                gem["pulse"] += 0.1

    def _create_particles(self, pos, color, count, speed_mult=1.0):
        for _ in range(count):
            angle = self.np_random.random() * 2 * math.pi
            speed = (self.np_random.random() * 2 + 1) * speed_mult
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = self.np_random.integers(15, 30)
            self.particles.append({"pos": list(pos), "vel": vel, "lifespan": lifespan, "color": color})

    def _update_particles(self):
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][1] += 0.1  # Gravity
            p["lifespan"] -= 1
        self.particles = [p for p in self.particles if p["lifespan"] > 0]

    def _render_game(self):
        # --- Draw grid ---
        for y in range(self.GRID_HEIGHT + 1):
            start = self._cart_to_iso(0, y)
            end = self._cart_to_iso(self.GRID_WIDTH, y)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end, 1)
        for x in range(self.GRID_WIDTH + 1):
            start = self._cart_to_iso(x, 0)
            end = self._cart_to_iso(x, self.GRID_HEIGHT)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end, 1)

        # --- Sort and draw entities ---
        entities = [{"e": self.player, "type": "player"}]
        for e in self.enemies: entities.append({"e": e, "type": "enemy"})
        for g in self.gems: entities.append({"e": g, "type": "gem"})
        
        # Sort by cartesian y-pos for correct isometric rendering
        entities.sort(key=lambda item: item["e"]["pos"][0] + item["e"]["pos"][1])

        for item in entities:
            entity = item["e"]
            vx, vy = entity["visual_pos"]
            
            if item["type"] == "gem":
                pulse_offset = math.sin(entity["pulse"]) * 2
                self._draw_iso_diamond(self.screen, self.COLOR_GEM, vx, vy - pulse_offset, self.COLOR_GEM_GLOW)
            elif item["type"] == "enemy":
                self._draw_iso_diamond(self.screen, self.COLOR_ENEMY, vx, vy, self.COLOR_ENEMY_GLOW)
            elif item["type"] == "player":
                # Blink when invulnerable
                if self.invulnerability_timer > 0 and (self.steps // 3) % 2 == 0:
                    continue
                self._draw_iso_diamond(self.screen, self.COLOR_PLAYER, vx, vy, self.COLOR_PLAYER_GLOW)

        # --- Draw particles ---
        for p in self.particles:
            size = max(1, p["lifespan"] / 6)
            pygame.draw.circle(self.screen, p["color"], p["pos"], size)

    def _render_ui(self):
        # Health display
        health_text = self.font_small.render(f"HEALTH: {self.health}", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (10, 10))

        # Score display
        score_text = self.font_small.render(f"GEMS: {self.score} / {self.GEMS_TO_WIN}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(score_text, score_rect)

        # Game Over / Win message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self.health <= 0:
                msg = "GAME OVER"
            elif self.score >= self.GEMS_TO_WIN:
                msg = "YOU WIN!"
            else:
                msg = "TIME'S UP!"
            
            end_text = self.font_large.render(msg, True, self.COLOR_TEXT)
            end_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, end_rect)


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
            "health": self.health,
            "gems_remaining": self.GEMS_TO_WIN - self.score
        }

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game directly
    # Requires pygame to be installed with display drivers
    # Re-enable display for human play
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # --- Human Controls ---
    # 0=none, 1=up, 2=down, 3=left, 4=right
    movement_key_map = {
        pygame.K_UP: 1,
        pygame.K_w: 1,
        pygame.K_DOWN: 2,
        pygame.K_s: 2,
        pygame.K_LEFT: 3,
        pygame.K_a: 3,
        pygame.K_RIGHT: 4,
        pygame.K_d: 4,
    }

    # Pygame setup for human play
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Isometric Gem Collector")
    clock = pygame.time.Clock()
    
    total_reward = 0.0

    while not done:
        # --- Action Mapping ---
        # Default action is no-op
        action = [0, 0, 0] 
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        keys = pygame.key.get_pressed()
        for key, move_action in movement_key_map.items():
            if keys[key]:
                action[0] = move_action
                break # Prioritize first key in map
        
        if keys[pygame.K_SPACE]:
            action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1

        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        # --- Rendering ---
        # The observation is already a rendered frame, so we just need to display it
        # Pygame uses (width, height), numpy uses (height, width)
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # --- Game Info ---
        print(f"Step: {info['steps']}, Score: {info['score']}, Health: {info['health']}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}")

        clock.tick(30) # Run at 30 FPS

    print("\n--- GAME OVER ---")
    print(f"Final Score: {info['score']}")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Total Steps: {info['steps']}")
    
    env.close()
    pygame.quit()