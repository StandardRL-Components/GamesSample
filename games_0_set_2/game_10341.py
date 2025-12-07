import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T10:13:49.424203
# Source Brief: brief_00341.md
# Brief Index: 341
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Defend the stationary humans from waves of incoming enemies. Use your momentum-based shockwave attack to push enemies away and cause them to collide with each other."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move. Press space to unleash a shockwave that pushes enemies away."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 2000

    PLAYER_SPEED = 5.0
    PLAYER_RADIUS = 15
    ENEMY_RADIUS = 10
    HUMAN_RADIUS = 12

    INITIAL_ENEMIES = 3
    INITIAL_HUMANS = 3
    HUMAN_MAX_HEALTH = 100

    ATTACK_COOLDOWN = 30  # steps
    ATTACK_SHOCKWAVE_DURATION = 15
    ATTACK_SHOCKWAVE_MAX_RADIUS = 120
    ATTACK_FORCE = 8.0

    PARTICLE_LIFESPAN = 15
    PARTICLE_SPEED = 3.0

    # Colors (Vibrant and Contrasting)
    COLOR_BG = (10, 20, 35)
    COLOR_BOUNDARY = (100, 100, 120)
    COLOR_PLAYER = (0, 150, 255)
    COLOR_PLAYER_GLOW = (0, 150, 255)
    COLOR_ENEMY = (255, 50, 50)
    COLOR_HUMAN = (50, 255, 100)
    COLOR_SPARK = (255, 255, 0)
    COLOR_TEXT = (220, 220, 240)
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Consolas", 24, bold=True)
        
        self.player_pos = np.zeros(2, dtype=np.float32)
        self.humans = []
        self.enemies = []
        self.particles = []
        self.shockwaves = []
        self.attack_cooldown_timer = 0
        self.base_enemy_speed = 0
        self.num_enemies_to_spawn = 0
        self.steps = 0
        self.score = 0
        self.game_over = False

        # self.reset() # reset is called by the environment wrapper
        # self.validate_implementation() # this is for debugging, should not be in __init__

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False

        self.player_pos = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2], dtype=np.float32)
        
        self._spawn_humans()
        
        self.enemies = []
        self.base_enemy_speed = 1.5
        self.num_enemies_to_spawn = self.INITIAL_ENEMIES
        for _ in range(self.num_enemies_to_spawn):
            self._spawn_enemy()

        self.particles = []
        self.shockwaves = []
        self.attack_cooldown_timer = 0
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        reward = 0.1  # Survival reward

        self._handle_player_input(movement, space_held)
        self._update_entities()
        
        collision_reward = self._handle_collisions()
        reward += collision_reward

        self._update_game_progression()
        
        self.steps += 1
        
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        
        if terminated and not truncated: # Lost all humans
             reward -= 10.0
        elif truncated and not terminated: # Survived until the end
             reward += 100.0
        
        self.score += reward
        self.game_over = terminated or truncated
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _spawn_humans(self):
        self.humans = []
        margin = 50
        positions = [
            (margin, self.SCREEN_HEIGHT / 2),
            (self.SCREEN_WIDTH - margin, self.SCREEN_HEIGHT / 3),
            (self.SCREEN_WIDTH - margin, 2 * self.SCREEN_HEIGHT / 3)
        ]
        for i in range(self.INITIAL_HUMANS):
            pos = positions[i % len(positions)]
            self.humans.append({
                "pos": np.array(pos, dtype=np.float32),
                "health": self.HUMAN_MAX_HEALTH,
                "radius": self.HUMAN_RADIUS
            })

    def _spawn_enemy(self):
        while True:
            edge = self.np_random.integers(4)
            if edge == 0: # top
                pos = np.array([self.np_random.uniform(0, self.SCREEN_WIDTH), -self.ENEMY_RADIUS], dtype=np.float32)
            elif edge == 1: # bottom
                pos = np.array([self.np_random.uniform(0, self.SCREEN_WIDTH), self.SCREEN_HEIGHT + self.ENEMY_RADIUS], dtype=np.float32)
            elif edge == 2: # left
                pos = np.array([-self.ENEMY_RADIUS, self.np_random.uniform(0, self.SCREEN_HEIGHT)], dtype=np.float32)
            else: # right
                pos = np.array([self.SCREEN_WIDTH + self.ENEMY_RADIUS, self.np_random.uniform(0, self.SCREEN_HEIGHT)], dtype=np.float32)

            # Ensure it's not spawning on top of the player
            if np.linalg.norm(pos - self.player_pos) > 100:
                break
        
        angle = self.np_random.uniform(0, 2 * math.pi)
        vel = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32) * self.base_enemy_speed
        self.enemies.append({"pos": pos, "vel": vel, "radius": self.ENEMY_RADIUS})

    def _handle_player_input(self, movement, space_held):
        if self.attack_cooldown_timer > 0:
            self.attack_cooldown_timer -= 1

        # Movement
        move_vec = np.zeros(2, dtype=np.float32)
        if movement == 1: move_vec[1] -= 1 # Up
        elif movement == 2: move_vec[1] += 1 # Down
        elif movement == 3: move_vec[0] -= 1 # Left
        elif movement == 4: move_vec[0] += 1 # Right
        
        if np.linalg.norm(move_vec) > 0:
            self.player_pos += move_vec * self.PLAYER_SPEED

        self.player_pos[0] = np.clip(self.player_pos[0], self.PLAYER_RADIUS, self.SCREEN_WIDTH - self.PLAYER_RADIUS)
        self.player_pos[1] = np.clip(self.player_pos[1], self.PLAYER_RADIUS, self.SCREEN_HEIGHT - self.PLAYER_RADIUS)

        # Momentum Attack
        if space_held and self.attack_cooldown_timer <= 0:
            # SFX: Player_Attack
            self.attack_cooldown_timer = self.ATTACK_COOLDOWN
            self.shockwaves.append({
                "pos": self.player_pos.copy(),
                "life": self.ATTACK_SHOCKWAVE_DURATION,
                "max_life": self.ATTACK_SHOCKWAVE_DURATION
            })
            for enemy in self.enemies:
                dist_vec = enemy["pos"] - self.player_pos
                distance = np.linalg.norm(dist_vec)
                if distance > 0 and distance < self.ATTACK_SHOCKWAVE_MAX_RADIUS:
                    force_mag = self.ATTACK_FORCE * (1 - (distance / self.ATTACK_SHOCKWAVE_MAX_RADIUS))
                    force_vec = (dist_vec / distance) * force_mag
                    enemy["vel"] += force_vec

    def _update_entities(self):
        # Enemies
        for enemy in self.enemies:
            enemy["pos"] += enemy["vel"]
            if enemy["pos"][0] <= enemy["radius"] or enemy["pos"][0] >= self.SCREEN_WIDTH - enemy["radius"]:
                enemy["vel"][0] *= -1
                enemy["pos"][0] = np.clip(enemy["pos"][0], enemy["radius"], self.SCREEN_WIDTH - enemy["radius"])
            if enemy["pos"][1] <= enemy["radius"] or enemy["pos"][1] >= self.SCREEN_HEIGHT - enemy["radius"]:
                enemy["vel"][1] *= -1
                enemy["pos"][1] = np.clip(enemy["pos"][1], enemy["radius"], self.SCREEN_HEIGHT - enemy["radius"])

        # Particles
        self.particles = [p for p in self.particles if p["life"] > 0]
        for p in self.particles:
            p["pos"] += p["vel"]
            p["life"] -= 1

        # Shockwaves
        self.shockwaves = [s for s in self.shockwaves if s["life"] > 0]
        for s in self.shockwaves:
            s["life"] -= 1

    def _handle_collisions(self):
        reward = 0
        # Enemy-Enemy
        for i in range(len(self.enemies)):
            for j in range(i + 1, len(self.enemies)):
                e1, e2 = self.enemies[i], self.enemies[j]
                dist_vec = e1["pos"] - e2["pos"]
                distance = np.linalg.norm(dist_vec)
                if distance > 0 and distance < e1["radius"] + e2["radius"]:
                    # SFX: Enemy_Collide
                    self._resolve_circle_collision(e1, e2, dist_vec, distance)
                    self._create_sparks(e1["pos"] + dist_vec * 0.5, 10)
                    reward += 1.0

        # Enemy-Human
        humans_to_remove = []
        for human in self.humans:
            for enemy in self.enemies:
                dist_vec = human["pos"] - enemy["pos"]
                distance = np.linalg.norm(dist_vec)
                if distance > 0 and distance < human["radius"] + enemy["radius"]:
                    # SFX: Human_Hit
                    damage = np.linalg.norm(enemy["vel"]) * 5
                    human["health"] -= damage
                    self._resolve_circle_collision(human, enemy, dist_vec, distance, human_is_static=True)
                    self._create_sparks(human["pos"] - dist_vec * (human["radius"] / distance), 5)
                    reward -= 0.5 # Penalty for hit
                    if human["health"] <= 0:
                        humans_to_remove.append(human)
                        # Penalty for elimination is now handled in step()
                        break
        
        if humans_to_remove:
            self.humans = [h for h in self.humans if h not in humans_to_remove]
        
        return reward

    def _resolve_circle_collision(self, c1, c2, dist_vec, distance, human_is_static=False):
        # Separate overlapping circles
        overlap = (c1["radius"] + c2["radius"] - distance)
        separation_vec = dist_vec / distance * overlap
        if not human_is_static:
            c1["pos"] += separation_vec * 0.5
            c2["pos"] -= separation_vec * 0.5
        else:
            c2["pos"] -= separation_vec

        # Elastic collision response
        normal = dist_vec / distance
        v1 = c1.get("vel", np.zeros(2))
        v2 = c2.get("vel", np.zeros(2))
        
        v_rel_normal = np.dot(v1 - v2, normal)
        if v_rel_normal < 0:
            impulse = (2.0 * v_rel_normal) / 2.0 # Assuming equal mass
            if not human_is_static:
                c1["vel"] -= impulse * normal
            c2["vel"] += impulse * normal

    def _create_sparks(self, pos, num_sparks):
        for _ in range(num_sparks):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(0.5, 1.0) * self.PARTICLE_SPEED
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            self.particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "life": self.np_random.integers(self.PARTICLE_LIFESPAN // 2, self.PARTICLE_LIFESPAN)
            })

    def _update_game_progression(self):
        if self.steps > 0:
            if self.steps % 100 == 0:
                self.base_enemy_speed += 0.05
            if self.steps % 500 == 0:
                self.num_enemies_to_spawn += 1
                self._spawn_enemy()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Boundary
        pygame.draw.rect(self.screen, self.COLOR_BOUNDARY, (0, 0, self.SCREEN_WIDTH, self.SCREEN_HEIGHT), 2)
        
        # Shockwaves
        for s in self.shockwaves:
            progress = (s["max_life"] - s["life"]) / s["max_life"]
            current_radius = int(progress * self.ATTACK_SHOCKWAVE_MAX_RADIUS)
            alpha = int(200 * (1 - progress))
            if current_radius > 0 and alpha > 0:
                # Create a temporary surface for the glowing circle
                temp_surf = pygame.Surface((current_radius * 2, current_radius * 2), pygame.SRCALPHA)
                # Draw the circle on the temporary surface
                pygame.draw.circle(temp_surf, (*self.COLOR_PLAYER, alpha), (current_radius, current_radius), current_radius, width=max(1, int(8 * (1-progress))))
                # Blit the temporary surface to the main screen
                self.screen.blit(temp_surf, (int(s["pos"][0] - current_radius), int(s["pos"][1] - current_radius)))

        # Humans
        for human in self.humans:
            pos = (int(human["pos"][0]), int(human["pos"][1]))
            health_ratio = max(0, human["health"] / self.HUMAN_MAX_HEALTH)
            color = (
                int(self.COLOR_HUMAN[0] * health_ratio + self.COLOR_ENEMY[0] * (1-health_ratio)),
                int(self.COLOR_HUMAN[1] * health_ratio),
                int(self.COLOR_HUMAN[2] * health_ratio)
            )
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], human["radius"], color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], human["radius"], color)

        # Enemies
        for enemy in self.enemies:
            pos = (int(enemy["pos"][0]), int(enemy["pos"][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], enemy["radius"], self.COLOR_ENEMY)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], enemy["radius"], self.COLOR_ENEMY)

        # Player
        self._render_glow_circle(self.screen, self.COLOR_PLAYER, self.player_pos, self.PLAYER_RADIUS, 5)

        # Particles
        for p in self.particles:
            alpha = 255 * (p["life"] / self.PARTICLE_LIFESPAN)
            size = int(3 * (p["life"] / self.PARTICLE_LIFESPAN))
            if size > 0:
                 # Create a temporary surface for alpha blending
                 temp_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                 pygame.draw.circle(temp_surf, (*self.COLOR_SPARK, alpha), (size, size), size)
                 self.screen.blit(temp_surf, (int(p["pos"][0] - size), int(p["pos"][1] - size)))


    def _render_glow_circle(self, surface, color, center, radius, strength):
        center_int = (int(center[0]), int(center[1]))
        for i in range(strength, 0, -1):
            alpha = int(100 * (1 - (i / strength)))
            glow_color = (*color[:3], alpha)
            glow_radius = radius + i
            temp_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, glow_color, (glow_radius, glow_radius), glow_radius)
            surface.blit(temp_surf, (center_int[0] - glow_radius, center_int[1] - glow_radius))

        pygame.gfxdraw.filled_circle(surface, center_int[0], center_int[1], radius, color)
        pygame.gfxdraw.aacircle(surface, center_int[0], center_int[1], radius, color)

    def _render_ui(self):
        time_left_text = self.font.render(f"TIME: {self.MAX_STEPS - self.steps}", True, self.COLOR_TEXT)
        self.screen.blit(time_left_text, (10, 10))

        humans_left_text = self.font.render(f"HUMANS: {len(self.humans)}/{self.INITIAL_HUMANS}", True, self.COLOR_TEXT)
        self.screen.blit(humans_left_text, (self.SCREEN_WIDTH - humans_left_text.get_width() - 10, 10))

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "humans_left": len(self.humans)}

    def _check_termination(self):
        return len(self.humans) == 0

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
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block is for manual testing and visualization.
    # It will not be executed by the evaluation server.
    # Set the video driver to a visible one, if not already set by the user.
    if os.environ.get("SDL_VIDEODRIVER", "dummy") == "dummy":
        os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "macOS"
    
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    render_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Momentum Defender")
    clock = pygame.time.Clock()
    
    total_reward = 0
    
    while not done:
        # --- Manual Control ---
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- RL Agent Control (uncomment to use) ---
        # action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        # Render to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        clock.tick(30) # Run at 30 FPS

    print(f"Game Over. Final Score: {info['score']:.2f}, Steps: {info['steps']}")
    env.close()