import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame



class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move. Hold Space to fire. Dodge enemy bullets and clear all waves to win."
    )

    game_description = (
        "A fast-paced, retro-arcade top-down shooter. Pilot your ship, blast through 5 waves of "
        "increasingly difficult aliens, and survive the onslaught to achieve victory."
    )

    auto_advance = True

    # --- Constants ---
    # Colors
    COLOR_BG = (15, 15, 35)
    COLOR_PLAYER = (50, 255, 50)
    COLOR_PLAYER_GLOW = (50, 255, 50, 50)
    COLOR_ENEMY_RED = (255, 50, 50)
    COLOR_ENEMY_BLUE = (50, 150, 255)
    COLOR_ENEMY_PURPLE = (200, 50, 255)
    COLOR_PLAYER_BULLET = (255, 255, 255)
    COLOR_ENEMY_BULLET = (255, 100, 200)
    COLOR_EXPLOSION = [(255, 200, 0), (255, 150, 0), (255, 100, 50)]
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_HEALTH_BAR = (0, 200, 0)
    COLOR_HEALTH_BAR_BG = (100, 0, 0)
    
    # Screen
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Game parameters
    MAX_STEPS = 3000 # Increased to allow for longer gameplay
    TOTAL_WAVES = 5
    ENEMIES_PER_WAVE = 20
    PLAYER_SPEED = 5
    PLAYER_HEALTH_MAX = 100
    PLAYER_SHOOT_COOLDOWN = 6  # frames
    PLAYER_BULLET_SPEED = 10
    
    # Entity sizes
    PLAYER_RADIUS = 12
    ENEMY_RADIUS = 10
    BULLET_RADIUS = 3

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Gymnasium spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        # Set dummy video driver for headless operation
        os.environ['SDL_VIDEODRIVER'] = 'dummy'
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 48, bold=True)
        
        # Internal state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.victory = False
        self.player_state = {}
        self.enemies = []
        self.player_projectiles = []
        self.enemy_projectiles = []
        self.particles = []
        self.stars = []
        self.current_wave = 0
        self.enemies_to_spawn = []
        self.spawn_timer = 0
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.victory = False
        
        self.player_state = {
            "pos": pygame.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 50),
            "health": self.PLAYER_HEALTH_MAX,
            "last_shot_time": 0,
            "aim_direction": pygame.Vector2(0, -1)
        }
        
        self.enemies.clear()
        self.player_projectiles.clear()
        self.enemy_projectiles.clear()
        self.particles.clear()
        
        self.current_wave = 0
        self._start_next_wave()

        if not self.stars:
            for _ in range(200):
                self.stars.append({
                    "pos": pygame.Vector2(self.np_random.uniform(0, self.SCREEN_WIDTH), self.np_random.uniform(0, self.SCREEN_HEIGHT)),
                    "speed": self.np_random.uniform(0.5, 2.0),
                    "size": self.np_random.integers(1, 4)
                })

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward_this_step = -0.01  # Small penalty for time passing to encourage efficiency
        
        if not self.game_over:
            self._handle_input(action)
            self._update_player()
            self._update_enemies()
            self._update_projectiles()
            
            reward_this_step += self._handle_collisions()
            
            self._update_spawner()
            
            if not self.enemies and not self.enemies_to_spawn:
                if self.current_wave < self.TOTAL_WAVES:
                    self._start_next_wave()
                    reward_this_step += 5 # Wave completion reward
                else:
                    self.victory = True
                    self.game_over = True
        
        self._update_particles()
        
        self.steps += 1
        
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        if terminated and not truncated: # Game ended due to win/loss
            if self.victory:
                reward_this_step += 50 # Victory reward
            else:
                reward_this_step -= 50 # Defeat penalty

        return (
            self._get_observation(),
            reward_this_step,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, _ = action
        
        move_vec = pygame.Vector2(0, 0)
        if movement == 1: move_vec.y = -1
        elif movement == 2: move_vec.y = 1
        elif movement == 3: move_vec.x = -1
        elif movement == 4: move_vec.x = 1
        
        if move_vec.length() > 0:
            move_vec.normalize_ip()
            self.player_state["pos"] += move_vec * self.PLAYER_SPEED
            self.player_state["aim_direction"] = move_vec
            self._clamp_player_position()

        if space_held and (self.steps - self.player_state["last_shot_time"] > self.PLAYER_SHOOT_COOLDOWN):
            # sfx: player_shoot
            self.player_state["last_shot_time"] = self.steps
            bullet_pos = self.player_state["pos"] + self.player_state["aim_direction"] * self.PLAYER_RADIUS
            self.player_projectiles.append({
                "pos": bullet_pos,
                "vel": self.player_state["aim_direction"] * self.PLAYER_BULLET_SPEED
            })

    def _clamp_player_position(self):
        self.player_state["pos"].x = max(self.PLAYER_RADIUS, min(self.SCREEN_WIDTH - self.PLAYER_RADIUS, self.player_state["pos"].x))
        self.player_state["pos"].y = max(self.PLAYER_RADIUS, min(self.SCREEN_HEIGHT - self.PLAYER_RADIUS, self.player_state["pos"].y))

    def _update_player(self):
        # Nothing to update here besides input handling
        pass

    def _update_enemies(self):
        enemy_projectile_speed = 4 + self.current_wave
        enemy_fire_rate_modifier = 1.0 - (self.current_wave * 0.1)

        for enemy in self.enemies:
            # Movement
            if enemy["type"] == 'red':
                enemy["pos"].x = enemy["spawn_x"] + math.sin(self.steps * 0.05 + enemy["phase"]) * 100
                enemy["pos"].y += 0.5
            elif enemy["type"] == 'blue':
                angle = self.steps * 0.03 + enemy["phase"]
                enemy["pos"].x = enemy["spawn_x"] + math.cos(angle) * 80
                enemy["pos"].y = enemy["spawn_y"] + math.sin(angle) * 80
            elif enemy["type"] == 'purple':
                target_x = self.player_state["pos"].x
                enemy["pos"].x += np.clip(target_x - enemy["pos"].x, -1.5, 1.5)
                enemy["pos"].y += 0.75

            # Firing
            if self.steps > enemy["last_shot_time"] + enemy["fire_rate"] * enemy_fire_rate_modifier:
                enemy["last_shot_time"] = self.steps
                # sfx: enemy_shoot
                if enemy["type"] == 'red':
                    self.enemy_projectiles.append({"pos": pygame.Vector2(enemy["pos"]), "vel": pygame.Vector2(0, enemy_projectile_speed)})
                elif enemy["type"] == 'blue':
                    for angle in [-0.5, 0, 0.5]:
                        vel = pygame.Vector2(0, enemy_projectile_speed).rotate_rad(angle)
                        self.enemy_projectiles.append({"pos": pygame.Vector2(enemy["pos"]), "vel": vel})
                elif enemy["type"] == 'purple':
                    direction = (self.player_state["pos"] - enemy["pos"]).normalize()
                    self.enemy_projectiles.append({"pos": pygame.Vector2(enemy["pos"]), "vel": direction * enemy_projectile_speed})

            # Despawn if off-screen or too old
            if enemy["pos"].y > self.SCREEN_HEIGHT + 20 or (self.steps - enemy["spawn_time"] > 1000):
                enemy["to_remove"] = True

    def _update_projectiles(self):
        for p in self.player_projectiles:
            p["pos"] += p["vel"]
        for p in self.enemy_projectiles:
            p["pos"] += p["vel"]

        # Remove off-screen projectiles
        self.player_projectiles = [p for p in self.player_projectiles if 0 < p["pos"].x < self.SCREEN_WIDTH and 0 < p["pos"].y < self.SCREEN_HEIGHT]
        self.enemy_projectiles = [p for p in self.enemy_projectiles if 0 < p["pos"].x < self.SCREEN_WIDTH and 0 < p["pos"].y < self.SCREEN_HEIGHT]

    def _handle_collisions(self):
        reward = 0
        
        # Player projectiles vs Enemies
        for p in self.player_projectiles:
            for e in self.enemies:
                if not e.get("to_remove", False) and (p["pos"] - e["pos"]).length() < self.ENEMY_RADIUS + self.BULLET_RADIUS:
                    p["to_remove"] = True
                    e["health"] -= 25
                    reward += 0.1 # Hit reward
                    if e["health"] <= 0:
                        e["to_remove"] = True
                        self.score += 100
                        reward += 1 # Kill reward
                        self._create_explosion(e["pos"], 20, e["color"])
                    else:
                        self._create_explosion(p["pos"], 5, e["color"]) # Hit spark
                    break
        
        # Enemy projectiles vs Player
        for p in self.enemy_projectiles:
            if not p.get("to_remove", False) and (p["pos"] - self.player_state["pos"]).length() < self.PLAYER_RADIUS + self.BULLET_RADIUS:
                p["to_remove"] = True
                self.player_state["health"] -= 10
                reward -= 0.1 # Damage penalty
                self._create_explosion(self.player_state["pos"], 10, self.COLOR_PLAYER)
                if self.player_state["health"] <= 0:
                    self.player_state["health"] = 0
                    self.game_over = True
                    self._create_explosion(self.player_state["pos"], 50, self.COLOR_PLAYER)
                break
        
        # Cleanup removed items
        self.player_projectiles = [p for p in self.player_projectiles if not p.get("to_remove", False)]
        self.enemies = [e for e in self.enemies if not e.get("to_remove", False)]
        self.enemy_projectiles = [p for p in self.enemy_projectiles if not p.get("to_remove", False)]
        
        return reward

    def _update_spawner(self):
        if self.enemies_to_spawn and self.steps > self.spawn_timer:
            self.enemies.append(self.enemies_to_spawn.pop(0))
            self.spawn_timer = self.steps + 20 # Spawn every 20 frames

    def _start_next_wave(self):
        self.current_wave += 1
        if self.current_wave > self.TOTAL_WAVES:
            return

        self.enemies_to_spawn.clear()
        for i in range(self.ENEMIES_PER_WAVE):
            enemy_type_roll = self.np_random.random()
            if self.current_wave == 1 or enemy_type_roll < 0.6:
                enemy_type = 'red'
                color = self.COLOR_ENEMY_RED
            elif self.current_wave < 4 and enemy_type_roll < 0.9:
                enemy_type = 'blue'
                color = self.COLOR_ENEMY_BLUE
            else:
                enemy_type = 'purple'
                color = self.COLOR_ENEMY_PURPLE

            spawn_x = self.np_random.uniform(100, self.SCREEN_WIDTH - 100)
            spawn_y = self.np_random.uniform(50, 150)
            
            self.enemies_to_spawn.append({
                "pos": pygame.Vector2(spawn_x, -20),
                "spawn_x": spawn_x,
                "spawn_y": spawn_y,
                "health": 100,
                "type": enemy_type,
                "color": color,
                "fire_rate": self.np_random.integers(80, 120),
                "last_shot_time": self.steps,
                "phase": self.np_random.uniform(0, 2 * math.pi),
                "spawn_time": self.steps
            })
        self.spawn_timer = self.steps + 60 # Initial delay before first spawn

    def _update_particles(self):
        for p in self.particles:
            p["pos"] += p["vel"]
            p["lifespan"] -= 1
        self.particles = [p for p in self.particles if p["lifespan"] > 0]

    def _create_explosion(self, pos, num_particles, base_color):
        # sfx: explosion
        for _ in range(num_particles):
            vel = pygame.Vector2(self.np_random.uniform(-3, 3), self.np_random.uniform(-3, 3))
            self.particles.append({
                "pos": pygame.Vector2(pos),
                "vel": vel,
                "lifespan": self.np_random.integers(10, 30),
                "color": random.choice(self.COLOR_EXPLOSION)
            })

    def _check_termination(self):
        return self.game_over

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_stars()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_stars(self):
        for star in self.stars:
            star["pos"].y = (star["pos"].y + star["speed"]) % self.SCREEN_HEIGHT
            color_val = int(100 + star["speed"] * 50)
            color = (color_val, color_val, color_val)
            pygame.draw.circle(self.screen, color, star["pos"], star["size"] / 2)

    def _render_game(self):
        # Render particles
        for p in self.particles:
            alpha = max(0, min(255, int(p["lifespan"] * 10)))
            p_color = (*p["color"], alpha)
            s = pygame.Surface((4, 4), pygame.SRCALPHA)
            pygame.draw.circle(s, p_color, (2, 2), 2)
            self.screen.blit(s, (int(p["pos"].x - 2), int(p["pos"].y - 2)))
        
        # Render projectiles
        for p in self.player_projectiles:
            pygame.gfxdraw.filled_circle(self.screen, int(p["pos"].x), int(p["pos"].y), self.BULLET_RADIUS, self.COLOR_PLAYER_BULLET)
        for p in self.enemy_projectiles:
            pygame.gfxdraw.filled_circle(self.screen, int(p["pos"].x), int(p["pos"].y), self.BULLET_RADIUS, self.COLOR_ENEMY_BULLET)
            pygame.gfxdraw.aacircle(self.screen, int(p["pos"].x), int(p["pos"].y), self.BULLET_RADIUS, self.COLOR_ENEMY_BULLET)

        # Render enemies
        for e in self.enemies:
            pos_int = (int(e["pos"].x), int(e["pos"].y))
            if e["type"] == 'red': # Diamond
                points = [(pos_int[0], pos_int[1] - self.ENEMY_RADIUS), (pos_int[0] + self.ENEMY_RADIUS, pos_int[1]),
                          (pos_int[0], pos_int[1] + self.ENEMY_RADIUS), (pos_int[0] - self.ENEMY_RADIUS, pos_int[1])]
                pygame.gfxdraw.aapolygon(self.screen, points, e["color"])
                pygame.gfxdraw.filled_polygon(self.screen, points, e["color"])
            elif e["type"] == 'blue': # Circle
                pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], self.ENEMY_RADIUS, e["color"])
                pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], self.ENEMY_RADIUS, e["color"])
            elif e["type"] == 'purple': # Hexagon
                points = []
                for i in range(6):
                    angle = math.pi / 3 * i
                    points.append((pos_int[0] + self.ENEMY_RADIUS * math.cos(angle), pos_int[1] + self.ENEMY_RADIUS * math.sin(angle)))
                pygame.gfxdraw.aapolygon(self.screen, points, e["color"])
                pygame.gfxdraw.filled_polygon(self.screen, points, e["color"])

        # Render player
        if self.player_state["health"] > 0:
            pos = self.player_state["pos"]
            # Glow effect
            glow_surf = pygame.Surface((self.PLAYER_RADIUS * 4, self.PLAYER_RADIUS * 4), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, self.COLOR_PLAYER_GLOW, (self.PLAYER_RADIUS*2, self.PLAYER_RADIUS*2), self.PLAYER_RADIUS*1.5)
            self.screen.blit(glow_surf, (int(pos.x - self.PLAYER_RADIUS*2), int(pos.y - self.PLAYER_RADIUS*2)))
            
            # Ship body (triangle)
            aim_angle = self.player_state["aim_direction"].angle_to(pygame.Vector2(0, -1))
            p1 = pos + pygame.Vector2(0, -self.PLAYER_RADIUS).rotate(-aim_angle)
            p2 = pos + pygame.Vector2(-self.PLAYER_RADIUS*0.8, self.PLAYER_RADIUS*0.8).rotate(-aim_angle)
            p3 = pos + pygame.Vector2(self.PLAYER_RADIUS*0.8, self.PLAYER_RADIUS*0.8).rotate(-aim_angle)
            points = [(p1.x, p1.y), (p2.x, p2.y), (p3.x, p3.y)]
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Wave
        wave_text = self.font_ui.render(f"WAVE: {self.current_wave}/{self.TOTAL_WAVES}", True, self.COLOR_UI_TEXT)
        self.screen.blit(wave_text, (self.SCREEN_WIDTH - wave_text.get_width() - 10, 10))

        # Health bar
        health_ratio = self.player_state["health"] / self.PLAYER_HEALTH_MAX
        bar_width = 200
        bar_height = 15
        bar_x = (self.SCREEN_WIDTH - bar_width) / 2
        bar_y = self.SCREEN_HEIGHT - bar_height - 10
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (bar_x, bar_y, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (bar_x, bar_y, bar_width * health_ratio, bar_height))

        # Game Over / Victory text
        if self.game_over:
            if self.victory:
                msg = "VICTORY"
                color = (100, 255, 100)
            else:
                msg = "GAME OVER"
                color = (255, 100, 100)
            
            text = self.font_game_over.render(msg, True, color)
            text_rect = text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.current_wave,
            "player_health": self.player_state["health"],
            "enemies_remaining": len(self.enemies) + len(self.enemies_to_spawn)
        }

    def close(self):
        pygame.quit()

# Example of how to run the environment
if __name__ == '__main__':
    import random # This was missing from the original __main__
    
    env = GameEnv(render_mode="rgb_array")
    
    # --- To run with manual control ---
    # This part requires a display. Comment out the os.environ line in __init__.
    # from gymnasium.utils.play import play
    # play(env, zoom=2, fps=30, keys_to_action={
    #     "w": np.array([1, 0, 0]), "s": np.array([2, 0, 0]),
    #     "a": np.array([3, 0, 0]), "d": np.array([4, 0, 0]),
    #     " ": np.array([0, 1, 0]),
    # })

    # --- To run a random agent ---
    print("Running random agent for one episode...")
    obs, info = env.reset()
    total_reward = 0
    terminated = False
    truncated = False
    step_count = 0
    while not terminated and not truncated:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1
        if step_count % 100 == 0:
            print(f"Step {step_count}: Info={info}, Last Reward={reward:.2f}")
    
    print(f"Episode finished after {step_count} steps. Final Info={info}")
    print(f"Total reward from random agent: {total_reward:.2f}")
    
    env.close()