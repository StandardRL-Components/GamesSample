
# Generated: 2025-08-28T04:37:12.835134
# Source Brief: brief_02373.md
# Brief Index: 2373

        
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
        "Controls: ↑↓←→ to move the placement cursor. Press space to build a tower."
    )

    game_description = (
        "Defend your base from waves of creeps by strategically placing towers in this minimalist top-down tower defense game."
    )

    auto_advance = True

    # --- CONSTANTS ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30 # Brief asks for 60, but 30 is better for simulation performance
    MAX_SECONDS = 60
    MAX_STEPS = MAX_SECONDS * FPS

    # Colors
    COLOR_BG = (25, 25, 35)
    COLOR_PATH = (50, 50, 65)
    COLOR_BASE_AREA = (40, 70, 40, 50) # RGBA for transparency
    COLOR_CREEP = (255, 50, 50)
    COLOR_TOWER = (50, 150, 255)
    COLOR_PROJECTILE = (255, 255, 255)
    COLOR_PARTICLE = (255, 220, 100)
    COLOR_TEXT = (230, 230, 230)
    COLOR_CURSOR_VALID = (50, 255, 50, 150)
    COLOR_CURSOR_INVALID = (255, 50, 50, 150)

    # Game Parameters
    CURSOR_SPEED = 10
    INITIAL_TOWERS = 10
    INITIAL_BASE_HEALTH = 10
    
    TOWER_SIZE = 12
    TOWER_RANGE = 80
    TOWER_COOLDOWN = 0.8 * FPS # seconds * FPS
    TOWER_PLACEMENT_BUFFER = 20

    CREEP_SIZE = 7
    CREEP_SPEED = 1.0
    CREEP_SPAWN_WAVE_INTERVAL = 10 * FPS # New wave every 10s
    CREEP_SPAWN_IN_WAVE_INTERVAL = 0.3 * FPS
    CREEP_WAVE_SIZE = 5
    DIFFICULTY_INTERVAL = 3 * CREEP_SPAWN_WAVE_INTERVAL # Health increases every 3 waves

    PROJECTILE_SPEED = 5
    PROJECTILE_DAMAGE = 1
    
    PATH_POINTS = [
        (-20, 200), (100, 200), (100, 100), (540, 100), (540, 300), (100, 300), (100, 200), (320, 200), (660, 200)
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_s = pygame.font.Font(None, 24)
        self.font_m = pygame.font.Font(None, 32)
        
        self.cursor_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=float)
        self.last_space_held = False

        self.creeps = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        
        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.creeps = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        
        self.cursor_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=float)
        self.last_space_held = False
        
        self.towers_available = self.INITIAL_TOWERS
        self.base_health = self.INITIAL_BASE_HEALTH
        
        self.creep_spawn_timer = self.CREEP_SPAWN_WAVE_INTERVAL - 120 # Start first wave sooner
        self.creeps_to_spawn = []
        self.creep_health_level = 1
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        
        self._handle_action(action)
        
        reward += self._update_spawner()
        reward += self._update_towers()
        reward += self._update_projectiles()
        reward += self._update_creeps()
        self._update_particles()
        
        self.steps += 1
        terminated = self._check_termination()
        
        if terminated and self.base_health > 0: # Win condition
            reward += 50.0

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_action(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        # Movement
        if movement == 1: self.cursor_pos[1] -= self.CURSOR_SPEED
        elif movement == 2: self.cursor_pos[1] += self.CURSOR_SPEED
        elif movement == 3: self.cursor_pos[0] -= self.CURSOR_SPEED
        elif movement == 4: self.cursor_pos[0] += self.CURSOR_SPEED
        
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.WIDTH)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.HEIGHT)
        
        # Tower Placement
        is_pressed = space_held and not self.last_space_held
        if is_pressed and self._is_placement_valid():
            self.towers_available -= 1
            self.towers.append({
                "pos": self.cursor_pos.copy(),
                "cooldown": 0,
            })
            # Sound: "place_tower.wav"
            for _ in range(15):
                self.particles.append(self._create_particle(self.cursor_pos, self.COLOR_TOWER, 2))

        self.last_space_held = space_held

    def _update_spawner(self):
        # Difficulty scaling
        if self.steps > 0 and self.steps % self.DIFFICULTY_INTERVAL == 0:
            self.creep_health_level += 1

        # Wave spawning
        self.creep_spawn_timer += 1
        if self.creep_spawn_timer >= self.CREEP_SPAWN_WAVE_INTERVAL:
            self.creep_spawn_timer = 0
            for i in range(self.CREEP_WAVE_SIZE):
                self.creeps_to_spawn.append({
                    "delay": i * self.CREEP_SPAWN_IN_WAVE_INTERVAL,
                    "health": self.creep_health_level
                })
        
        # Individual creep spawning from wave queue
        if self.creeps_to_spawn:
            self.creeps_to_spawn[0]["delay"] -= 1
            if self.creeps_to_spawn[0]["delay"] <= 0:
                spawn_info = self.creeps_to_spawn.pop(0)
                self.creeps.append({
                    "pos": np.array(self.PATH_POINTS[0], dtype=float),
                    "health": spawn_info["health"],
                    "max_health": spawn_info["health"],
                    "waypoint_idx": 1,
                })
        return 0

    def _update_towers(self):
        for tower in self.towers:
            tower["cooldown"] = max(0, tower["cooldown"] - 1)
            if tower["cooldown"] > 0:
                continue

            target = None
            min_dist = self.TOWER_RANGE
            
            for creep in self.creeps:
                dist = np.linalg.norm(creep["pos"] - tower["pos"])
                if dist < min_dist:
                    min_dist = dist
                    target = creep
            
            if target:
                tower["cooldown"] = self.TOWER_COOLDOWN
                direction = (target["pos"] - tower["pos"])
                if np.linalg.norm(direction) > 0:
                    direction = direction / np.linalg.norm(direction)
                
                self.projectiles.append({
                    "pos": tower["pos"].copy(),
                    "vel": direction * self.PROJECTILE_SPEED,
                })
                # Sound: "tower_shoot.wav"
        return 0

    def _update_projectiles(self):
        reward = 0
        projectiles_to_keep = []
        for proj in self.projectiles:
            proj["pos"] += proj["vel"]
            
            hit = False
            for creep in self.creeps:
                if np.linalg.norm(proj["pos"] - creep["pos"]) < self.CREEP_SIZE:
                    creep["health"] -= self.PROJECTILE_DAMAGE
                    reward += 0.1 # Reward for hit
                    hit = True
                    # Sound: "projectile_hit.wav"
                    for _ in range(5):
                        self.particles.append(self._create_particle(proj["pos"], self.COLOR_PARTICLE, 1.5))
                    break
            
            if not hit and 0 < proj["pos"][0] < self.WIDTH and 0 < proj["pos"][1] < self.HEIGHT:
                projectiles_to_keep.append(proj)
        
        self.projectiles = projectiles_to_keep
        return reward

    def _update_creeps(self):
        reward = 0
        creeps_to_keep = []
        for creep in self.creeps:
            if creep["health"] <= 0:
                self.score += 1
                reward += 1.0 # Reward for kill
                # Sound: "creep_destroy.wav"
                for _ in range(20):
                    self.particles.append(self._create_particle(creep["pos"], self.COLOR_CREEP, 3))
                continue
            
            if creep["waypoint_idx"] >= len(self.PATH_POINTS):
                self.base_health -= 1
                reward -= 5.0 # Penalty for leak
                # Sound: "base_damage.wav"
                continue

            target_pos = np.array(self.PATH_POINTS[creep["waypoint_idx"]], dtype=float)
            direction = target_pos - creep["pos"]
            dist = np.linalg.norm(direction)
            
            if dist < self.CREEP_SPEED:
                creep["waypoint_idx"] += 1
            else:
                creep["pos"] += (direction / dist) * self.CREEP_SPEED
            
            creeps_to_keep.append(creep)
        
        self.creeps = creeps_to_keep
        return reward
        
    def _update_particles(self):
        self.particles = [p for p in self.particles if p["life"] > 0]
        for p in self.particles:
            p["pos"] += p["vel"]
            p["life"] -= 1
            p["size"] = max(0, p["size"] - 0.1)

    def _check_termination(self):
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
        if self.base_health <= 0:
            self.game_over = True
        return self.game_over

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "base_health": self.base_health}

    def _render_game(self):
        # Path
        for i in range(len(self.PATH_POINTS) - 1):
            p1 = self.PATH_POINTS[i]
            p2 = self.PATH_POINTS[i+1]
            pygame.draw.line(self.screen, self.COLOR_PATH, p1, p2, 30)
            pygame.draw.circle(self.screen, self.COLOR_PATH, p1, 15)
            pygame.draw.circle(self.screen, self.COLOR_PATH, p2, 15)

        # Towers
        for tower in self.towers:
            pos = (int(tower["pos"][0]), int(tower["pos"][1]))
            pygame.draw.rect(self.screen, self.COLOR_TOWER, (pos[0] - self.TOWER_SIZE//2, pos[1] - self.TOWER_SIZE//2, self.TOWER_SIZE, self.TOWER_SIZE), border_radius=2)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.TOWER_RANGE, (*self.COLOR_TOWER, 50))

        # Projectiles
        for proj in self.projectiles:
            start_pos = (int(proj["pos"][0]), int(proj["pos"][1]))
            end_pos = (int(proj["pos"][0] - proj["vel"][0]*1.5), int(proj["pos"][1] - proj["vel"][1]*1.5))
            pygame.draw.aaline(self.screen, self.COLOR_PROJECTILE, start_pos, end_pos, 2)

        # Creeps
        for creep in self.creeps:
            pos = (int(creep["pos"][0]), int(creep["pos"][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.CREEP_SIZE, self.COLOR_CREEP)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.CREEP_SIZE, self.COLOR_CREEP)
            # Health bar
            health_ratio = creep["health"] / creep["max_health"]
            bar_width = self.CREEP_SIZE * 2
            bar_height = 4
            bar_x = pos[0] - bar_width / 2
            bar_y = pos[1] - self.CREEP_SIZE - 6
            pygame.draw.rect(self.screen, (80,0,0), (bar_x, bar_y, bar_width, bar_height))
            pygame.draw.rect(self.screen, (0,200,0), (bar_x, bar_y, bar_width * health_ratio, bar_height))

        # Particles
        for p in self.particles:
            pos = (int(p["pos"][0]), int(p["pos"][1]))
            size = int(p["size"])
            if size > 0:
                alpha = int(255 * (p["life"] / p["max_life"]))
                color = (*p["color"], alpha)
                s = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.circle(s, color, (size, size), size)
                self.screen.blit(s, (pos[0]-size, pos[1]-size))
        
        # Cursor
        cursor_surf = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        pos = (int(self.cursor_pos[0]), int(self.cursor_pos[1]))
        color = self.COLOR_CURSOR_VALID if self._is_placement_valid() else self.COLOR_CURSOR_INVALID
        pygame.gfxdraw.filled_circle(cursor_surf, pos[0], pos[1], self.TOWER_SIZE, color)
        pygame.gfxdraw.aacircle(cursor_surf, pos[0], pos[1], self.TOWER_SIZE, color)
        pygame.gfxdraw.aacircle(cursor_surf, pos[0], pos[1], self.TOWER_RANGE, (*color[:3], 80))
        self.screen.blit(cursor_surf, (0,0))

    def _render_ui(self):
        # Timer
        time_left = max(0, self.MAX_SECONDS - self.steps / self.FPS)
        timer_text = f"TIME: {time_left:.1f}"
        text_surf = self.font_m.render(timer_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surf, (10, 10))
        
        # Score
        score_text = f"KILLS: {self.score}"
        text_surf = self.font_m.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surf, (self.WIDTH - text_surf.get_width() - 10, 10))

        # Towers Available
        towers_text = f"TOWERS: {self.towers_available}"
        text_surf = self.font_m.render(towers_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surf, (self.WIDTH - text_surf.get_width() - 10, self.HEIGHT - text_surf.get_height() - 10))
        
        # Base Health
        health_text = f"HEALTH: {self.base_health}"
        text_surf = self.font_m.render(health_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surf, (10, self.HEIGHT - text_surf.get_height() - 10))

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0,0))
            
            status_text = "VICTORY!" if self.base_health > 0 else "GAME OVER"
            status_surf = self.font_m.render(status_text, True, self.COLOR_TEXT)
            self.screen.blit(status_surf, (self.WIDTH/2 - status_surf.get_width()/2, self.HEIGHT/2 - status_surf.get_height()))
            
            score_surf = self.font_s.render(f"Final Kills: {self.score}", True, self.COLOR_TEXT)
            self.screen.blit(score_surf, (self.WIDTH/2 - score_surf.get_width()/2, self.HEIGHT/2 + 10))

    def _is_placement_valid(self):
        if self.towers_available <= 0:
            return False
        
        # Check distance to other towers
        for tower in self.towers:
            if np.linalg.norm(self.cursor_pos - tower["pos"]) < self.TOWER_PLACEMENT_BUFFER:
                return False
        
        # Check distance to path
        for i in range(len(self.PATH_POINTS) - 1):
            p1 = np.array(self.PATH_POINTS[i])
            p2 = np.array(self.PATH_POINTS[i+1])
            l2 = np.sum((p1 - p2)**2)
            if l2 == 0.0:
                if np.linalg.norm(self.cursor_pos - p1) < self.TOWER_PLACEMENT_BUFFER:
                    return False
            else:
                t = max(0, min(1, np.dot(self.cursor_pos - p1, p2 - p1) / l2))
                projection = p1 + t * (p2 - p1)
                if np.linalg.norm(self.cursor_pos - projection) < self.TOWER_PLACEMENT_BUFFER:
                    return False
        return True

    def _create_particle(self, pos, color, speed_mult):
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(1, 3) * speed_mult
        vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
        life = random.randint(10, 20)
        return {
            "pos": pos.copy(),
            "vel": vel,
            "size": random.uniform(2, 5),
            "life": life,
            "max_life": life,
            "color": color,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Tower Defense")
    
    running = True
    while running:
        action = [0, 0, 0] # no-op, released, released
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
            
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()

        obs, reward, terminated, truncated, info = env.step(action)
        
        # Transpose back for pygame display
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}")
            pygame.time.wait(2000) # Pause for 2 seconds
            obs, info = env.reset()

    env.close()