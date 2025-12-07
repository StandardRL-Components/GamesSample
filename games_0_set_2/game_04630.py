
# Generated: 2025-08-28T02:58:52.712245
# Source Brief: brief_04630.md
# Brief Index: 4630

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


# Helper class for particles (explosions, mining effects)
class Particle:
    def __init__(self, x, y, vx, vy, color, size, lifespan, gravity=0):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.color = color
        self.size = size
        self.lifespan = lifespan
        self.initial_lifespan = lifespan
        self.gravity = gravity

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += self.gravity
        self.lifespan -= 1
        return self.lifespan > 0

    def draw(self, surface, iso_converter):
        if self.lifespan <= 0:
            return
        
        screen_pos = iso_converter(self.x, self.y)
        current_size = max(0, int(self.size * (self.lifespan / self.initial_lifespan)))
        
        alpha = int(255 * (self.lifespan / self.initial_lifespan))
        r, g, b = self.color
        
        # Create a temporary surface for alpha blending
        temp_surf = pygame.Surface((current_size * 2, current_size * 2), pygame.SRCALPHA)
        pygame.draw.circle(temp_surf, (r, g, b, alpha), (current_size, current_size), current_size)
        surface.blit(temp_surf, (screen_pos[0] - current_size, screen_pos[1] - current_size))

# Main Game Environment
class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move. Hold Space while near an asteroid to mine it for ore."
    )
    game_description = (
        "Pilot a mining ship in a dangerous asteroid field. Collect 100 units of ore while dodging deadly energy beams to win."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    WIN_ORE = 100
    MAX_STEPS = 5000
    INITIAL_LIVES = 3
    PLAYER_SPEED = 3
    MINING_RANGE = 60
    MINING_RATE = 0.2
    
    # Colors
    COLOR_BG = (15, 10, 30)
    COLOR_PLAYER = (0, 150, 255)
    COLOR_PLAYER_GLOW = (0, 100, 200)
    COLOR_LASER = (255, 50, 50)
    COLOR_ASTEROID = (90, 90, 90)
    COLOR_ORE_VEINS = {
        "gold": (255, 215, 0),
        "silver": (192, 192, 192),
        "bronze": (205, 127, 50)
    }
    COLOR_TEXT = (220, 220, 220)
    COLOR_WIN = (100, 255, 100)
    COLOR_LOSE = (255, 100, 100)

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
        self.font_small = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)
        
        self.game_world_size = 300 # Logical radius of the play area

        # Pre-generate starfield
        self.stars = []
        for _ in range(150):
            x = random.randint(-self.game_world_size * 2, self.game_world_size * 2)
            y = random.randint(-self.game_world_size * 2, self.game_world_size * 2)
            size = random.uniform(0.5, 1.5)
            brightness = random.randint(50, 120)
            self.stars.append((x, y, size, (brightness, brightness, brightness)))

        # This attribute is used to decide whether to draw the mining beam
        self.is_mining_this_step = False
        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Player state
        self.player_pos = np.array([0.0, 0.0])
        self.player_lives = self.INITIAL_LIVES
        self.player_invincible_timer = 0
        
        # Game state
        self.steps = 0
        self.score = 0 # Ore collected
        self.game_over = False
        self.win = False
        
        # Entities
        self.asteroids = [self._create_asteroid() for _ in range(5)]
        self.lasers = []
        self.particles = []
        
        # Difficulty scaling
        self.laser_spawn_timer = 0
        self.laser_base_freq = 0.5 # lasers per second
        
        # Reward calculation state
        self.dist_to_closest_asteroid = self._get_dist_to_closest_asteroid()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(30) # Maintain 30 FPS
            
        reward = 0
        self.is_mining_this_step = False
        
        # --- 1. Handle Actions & Update Player ---
        if not self.game_over:
            movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
            
            dist_before = self._get_dist_to_closest_asteroid()

            # Update player position
            move_vec = np.array([0.0, 0.0])
            if movement == 1: move_vec[1] -= 1 # Up
            elif movement == 2: move_vec[1] += 1 # Down
            elif movement == 3: move_vec[0] -= 1 # Left
            elif movement == 4: move_vec[0] += 1 # Right
            
            if np.linalg.norm(move_vec) > 0:
                move_vec = move_vec / np.linalg.norm(move_vec)
            
            self.player_pos += move_vec * self.PLAYER_SPEED
            self.player_pos = np.clip(self.player_pos, -self.game_world_size, self.game_world_size)

            dist_after = self._get_dist_to_closest_asteroid()
            if dist_after < dist_before:
                reward += 0.1
            elif dist_after > dist_before:
                reward -= 0.01
            self.dist_to_closest_asteroid = dist_after

            # Mining action
            if space_held:
                mined_ore = self._handle_mining()
                if mined_ore > 0:
                    self.is_mining_this_step = True
                    self.score += mined_ore
                    reward += mined_ore * 1.0
                    # sfx: mining_success.wav

        # --- 2. Update Game World ---
        self._update_lasers()
        self._update_asteroids()
        self._update_particles()
        
        if not self.game_over:
            if self.player_invincible_timer <= 0:
                for laser in self.lasers:
                    dist = np.linalg.norm(self.player_pos - laser["pos"])
                    if dist < 10: # Collision radius
                        self._player_hit()
                        reward -= 10
                        self.lasers.remove(laser)
                        break
            else:
                self.player_invincible_timer -= 1
        
            laser_freq = self.laser_base_freq + (self.steps / 30) * 0.01
            self.laser_spawn_timer += 1 / 30.0
            if self.laser_spawn_timer > 1.0 / laser_freq:
                self.laser_spawn_timer = 0
                self._spawn_laser()
                # sfx: laser_spawn.wav

        self.steps += 1
        
        # --- 3. Check Termination ---
        terminated = False
        if self.score >= self.WIN_ORE:
            self.score = self.WIN_ORE
            self.win = True
            self.game_over = True
            terminated = True
            reward += 100
        elif self.player_lives <= 0:
            self.game_over = True
            terminated = True
        elif self.steps >= self.MAX_STEPS:
            self.game_over = True
            terminated = True
            
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _iso_to_screen(self, x, y):
        screen_x = self.SCREEN_WIDTH / 2 + (x - y) * 1.2
        screen_y = self.SCREEN_HEIGHT / 2 + (x + y) * 0.6
        return int(screen_x), int(screen_y)

    def _create_asteroid(self):
        angle = self.np_random.uniform(0, 2 * math.pi)
        dist = self.np_random.uniform(self.game_world_size * 0.2, self.game_world_size * 0.9)
        pos = np.array([math.cos(angle) * dist, math.sin(angle) * dist])
        
        size = self.np_random.uniform(15, 30)
        ore_type = self.np_random.choice(list(self.COLOR_ORE_VEINS.keys()))
        ore_content = int(size * self.np_random.uniform(0.8, 1.2))
        
        num_points = 8
        points = []
        for i in range(num_points):
            a = 2 * math.pi * i / num_points
            r = size * self.np_random.uniform(0.8, 1.2)
            points.append((r * math.cos(a), r * math.sin(a)))
            
        return {
            "pos": pos, "size": size, "ore_content": ore_content, "initial_ore": ore_content,
            "ore_type": ore_type, "shape_points": points, "respawn_timer": 0
        }

    def _handle_mining(self):
        closest_ast, dist = self._get_closest_asteroid()
        mined_ore = 0
        if closest_ast and dist < self.MINING_RANGE and closest_ast["ore_content"] > 0:
            # sfx: mining_beam.wav
            ore_to_mine = self.MINING_RATE
            mined_ore = min(closest_ast["ore_content"], ore_to_mine)
            closest_ast["ore_content"] -= mined_ore
            
            if self.steps % 3 == 0:
                p_angle = self.np_random.uniform(0, 2 * math.pi)
                p_vel = np.array([math.cos(p_angle), math.sin(p_angle)]) * 0.5
                self.particles.append(Particle(
                    closest_ast["pos"][0], closest_ast["pos"][1],
                    p_vel[0], p_vel[1], self.COLOR_ORE_VEINS[closest_ast["ore_type"]],
                    3, 60
                ))
        return mined_ore
        
    def _player_hit(self):
        # sfx: explosion.wav
        self.player_lives -= 1
        self.player_invincible_timer = 90
        for _ in range(50):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            color = self.np_random.choice([(255, 50, 50), (255, 150, 0), (255, 255, 100)])
            self.particles.append(Particle(
                self.player_pos[0], self.player_pos[1], vx, vy, color,
                self.np_random.uniform(2, 5), self.np_random.integers(30, 60), gravity=0.05
            ))

    def _spawn_laser(self):
        edge = self.np_random.integers(0, 4)
        w, h = self.game_world_size, self.game_world_size
        if edge == 0: start_pos = np.array([self.np_random.uniform(-w, w), -h - 20])
        elif edge == 1: start_pos = np.array([self.np_random.uniform(-w, w), h + 20])
        elif edge == 2: start_pos = np.array([-w - 20, self.np_random.uniform(-h, h)])
        else: start_pos = np.array([w + 20, self.np_random.uniform(-h, h)])
        
        target_pos = np.array([self.np_random.uniform(-w*0.5, w*0.5), self.np_random.uniform(-h*0.5, h*0.5)])
        direction = (target_pos - start_pos) / np.linalg.norm(target_pos - start_pos)
        speed = self.np_random.uniform(3, 5)
        
        self.lasers.append({"pos": start_pos, "vel": direction * speed})

    def _update_lasers(self):
        for laser in self.lasers[:]:
            laser["pos"] += laser["vel"]
            x, y = laser["pos"]
            if not (-self.game_world_size-30 < x < self.game_world_size+30 and \
                    -self.game_world_size-30 < y < self.game_world_size+30):
                self.lasers.remove(laser)
    
    def _update_asteroids(self):
        for ast in self.asteroids:
            if ast["ore_content"] <= 0 and ast["respawn_timer"] == 0:
                ast["respawn_timer"] = self.np_random.integers(300, 600)
            
            if ast["respawn_timer"] > 0:
                ast["respawn_timer"] -= 1
                if ast["respawn_timer"] == 0:
                    new_ast = self._create_asteroid()
                    ast.update(new_ast)

    def _update_particles(self):
        self.particles = [p for p in self.particles if p.update()]

    def _get_closest_asteroid(self):
        active_asteroids = [a for a in self.asteroids if a["ore_content"] > 0]
        if not active_asteroids:
            return None, float('inf')
        
        dists = [np.linalg.norm(self.player_pos - ast["pos"]) for ast in active_asteroids]
        min_idx = np.argmin(dists)
        return active_asteroids[min_idx], dists[min_idx]

    def _get_dist_to_closest_asteroid(self):
        _, dist = self._get_closest_asteroid()
        return dist

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        offset_x, offset_y = self.player_pos
        
        for x, y, size, color in self.stars:
            sx, sy = self._iso_to_screen(x - offset_x, y - offset_y)
            pygame.draw.circle(self.screen, color, (sx, sy), size)

        render_queue = []
        render_queue.extend(self.asteroids)
        render_queue.append({"type": "player", "pos": self.player_pos})
        render_queue.extend([p for p in self.particles])
        
        render_queue.sort(key=lambda obj: obj["pos"][1] if isinstance(obj, dict) else obj.y)

        for obj in render_queue:
            if isinstance(obj, Particle):
                obj.draw(self.screen, lambda x, y: self._iso_to_screen(x - offset_x, y - offset_y))
            elif "type" in obj and obj["type"] == "player":
                self._render_player()
            else: # Asteroid
                self._render_asteroid(obj, offset_x, offset_y)

        self._render_lasers(offset_x, offset_y)
        self._render_mining_beam(offset_x, offset_y)
        
    def _render_player(self):
        if self.player_lives <= 0: return

        screen_pos = (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2)

        if self.player_invincible_timer > 0 and self.steps % 10 < 5:
            return

        ship_points = [(-8, 10), (0, -12), (8, 10), (0, 5)]
        rotated_points = [pygame.math.Vector2(p).rotate(0) for p in ship_points]
        screen_points = [(p.x + screen_pos[0], p.y + screen_pos[1]) for p in rotated_points]
        
        pygame.gfxdraw.filled_polygon(self.screen, screen_points, self.COLOR_PLAYER)
        pygame.gfxdraw.aapolygon(self.screen, screen_points, self.COLOR_PLAYER)
        
        pygame.gfxdraw.filled_circle(self.screen, screen_pos[0], screen_pos[1] + 2, 12, (*self.COLOR_PLAYER_GLOW, 50))

    def _render_asteroid(self, asteroid, offset_x, offset_y):
        if asteroid["respawn_timer"] > 0: return

        sx, sy = self._iso_to_screen(asteroid["pos"][0] - offset_x, asteroid["pos"][1] - offset_y)
        
        shape = [(p[0] + sx, p[1] + sy) for p in asteroid["shape_points"]]
        pygame.gfxdraw.filled_polygon(self.screen, shape, self.COLOR_ASTEROID)
        pygame.gfxdraw.aapolygon(self.screen, shape, tuple(c*0.8 for c in self.COLOR_ASTEROID))

        if asteroid["ore_content"] > 0:
            ore_color = self.COLOR_ORE_VEINS[asteroid["ore_type"]]
            ore_ratio = asteroid["ore_content"] / asteroid["initial_ore"]
            ore_shape = [(p[0] * ore_ratio + sx, p[1] * ore_ratio + sy) for p in asteroid["shape_points"]]
            pygame.gfxdraw.filled_polygon(self.screen, ore_shape, ore_color)
            pygame.gfxdraw.aapolygon(self.screen, ore_shape, tuple(c*0.8 for c in ore_color))

    def _render_lasers(self, offset_x, offset_y):
        for laser in self.lasers:
            start_iso = laser["pos"] - laser["vel"] * 0.5
            end_iso = laser["pos"] + laser["vel"] * 0.5
            start_screen = self._iso_to_screen(start_iso[0] - offset_x, start_iso[1] - offset_y)
            end_screen = self._iso_to_screen(end_iso[0] - offset_x, end_iso[1] - offset_y)
            
            pygame.draw.aaline(self.screen, self.COLOR_LASER, start_screen, end_screen, 2)

    def _render_mining_beam(self, offset_x, offset_y):
        if self.is_mining_this_step:
            closest_ast, _ = self._get_closest_asteroid()
            if not closest_ast: return

            player_screen_pos = (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2)
            ast_screen_pos = self._iso_to_screen(closest_ast["pos"][0] - offset_x, closest_ast["pos"][1] - offset_y)
            
            t = self.steps % 20 / 20.0
            color = (int(100 + 155 * t), int(200 + 55 * t), 255)
            pygame.draw.aaline(self.screen, color, player_screen_pos, ast_screen_pos)

    def _render_ui(self):
        ore_text = self.font_small.render(f"ORE: {int(self.score)} / {self.WIN_ORE}", True, self.COLOR_TEXT)
        self.screen.blit(ore_text, (10, 10))
        
        for i in range(self.player_lives):
            ship_points = [(-5, 6), (0, -7), (5, 6)]
            screen_points = [(p[0] + self.SCREEN_WIDTH - 30 - i * 20, p[1] + 20) for p in ship_points]
            pygame.gfxdraw.filled_polygon(self.screen, screen_points, self.COLOR_PLAYER)

        if self.game_over:
            msg = "MISSION COMPLETE" if self.win else "GAME OVER"
            color = self.COLOR_WIN if self.win else self.COLOR_LOSE
            
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.player_lives,
            "laser_freq": self.laser_base_freq + (self.steps / 30) * 0.01
        }
    
    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    import os
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    
    env = GameEnv(render_mode="rgb_array")
    
    pygame.display.init()
    pygame.display.set_caption("Asteroid Miner")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    obs, info = env.reset()
    done = False
    
    while not done:
        keys = pygame.key.get_pressed()
        movement = 0
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
    
    if env.game_over:
        pygame.time.wait(2000)

    env.close()