
# Generated: 2025-08-27T20:26:26.277210
# Source Brief: brief_02460.md
# Brief Index: 2460

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ↑↓←→ to move. Hold Shift for a short speed boost. Press Space to fire."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Pilot a spaceship in a top-down shooter, blasting through 5 waves of aliens to survive and achieve the highest score."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        
        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        
        # Fonts
        self.font_ui = pygame.font.Font(None, 24)
        self.font_game_over = pygame.font.Font(None, 72)

        # Colors
        self.COLOR_BG = (10, 15, 30)
        self.COLOR_PLAYER = (0, 150, 255)
        self.COLOR_PLAYER_BOOST = (100, 200, 255)
        self.COLOR_PROJECTILE_PLAYER = (255, 255, 255)
        self.COLOR_PROJECTILE_ENEMY = (255, 100, 100)
        self.WAVE_COLORS = [
            (255, 50, 50),   # Wave 1: Red
            (50, 255, 50),   # Wave 2: Green
            (255, 255, 50),  # Wave 3: Yellow
            (200, 50, 255),  # Wave 4: Purple
            (255, 150, 50)   # Wave 5: Orange
        ]
        self.HEALTH_COLORS = [
            (255, 0, 100),   # 1 HP: Red/Pink
            (0, 200, 255),   # 2 HP: Cyan
            (0, 150, 255),   # 3 HP: Blue
        ]

        # Game constants
        self.MAX_STEPS = 5000
        self.PLAYER_MAX_HEALTH = 3
        self.PLAYER_SPEED = 4.0
        self.BOOST_DURATION = 5  # steps
        self.BOOST_MULTIPLIER = 2.0
        self.PROJECTILE_SPEED = 10.0
        self.ENEMIES_PER_WAVE = 20
        self.MAX_WAVES = 5
        self.INACTION_THRESHOLD = 5 # steps

        # Initialize state variables
        self.stars = []
        self.player_pos = [0,0]
        self.player_health = 0
        self.boost_timer = 0
        self.projectiles = []
        self.enemies = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.current_wave = 0
        self.game_over = False
        self.game_won = False
        self.last_space_held = False
        self.inaction_counter = 0

        self.reset()
        
        # Run validation check
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        
        self.player_pos = [self.WIDTH / 2, self.HEIGHT - 50]
        self.player_health = self.PLAYER_MAX_HEALTH
        self.boost_timer = 0
        
        self.projectiles = []
        self.enemies = []
        self.particles = []
        
        self.last_space_held = False
        self.inaction_counter = 0
        
        self.current_wave = 0
        self._start_next_wave()

        # Generate a static starfield
        self.stars = []
        for _ in range(150):
            self.stars.append({
                "pos": [self.np_random.uniform(0, self.WIDTH), self.np_random.uniform(0, self.HEIGHT)],
                "size": self.np_random.uniform(0.5, 1.5),
                "color": self.np_random.integers(50, 120)
            })
        
        return self._get_observation(), self._get_info()

    def _start_next_wave(self):
        self.current_wave += 1
        if self.current_wave > self.MAX_WAVES:
            self.game_won = True
            self.game_over = True
            return

        self.enemies.clear()
        wave_index = self.current_wave - 1
        enemy_speed = 1.0 + wave_index * 0.5
        
        for i in range(self.ENEMIES_PER_WAVE):
            if self.current_wave == 1: # Horizontal
                pos = [self.np_random.uniform(50, self.WIDTH - 50), self.np_random.uniform(30, 100)]
                vel = [enemy_speed * self.np_random.choice([-1, 1]), 0]
            elif self.current_wave == 2: # Vertical
                pos = [self.np_random.uniform(30, self.WIDTH - 30), self.np_random.uniform(50, 150)]
                vel = [0, enemy_speed * self.np_random.choice([-1, 1])]
            elif self.current_wave == 3: # Diagonal
                pos = [self.np_random.uniform(50, self.WIDTH - 50), self.np_random.uniform(30, 100)]
                vel = [enemy_speed * self.np_random.choice([-1, 1]), enemy_speed * self.np_random.choice([-1, 1])]
            elif self.current_wave == 4: # Circular
                center = [self.np_random.uniform(100, self.WIDTH - 100), self.np_random.uniform(100, 200)]
                radius = self.np_random.uniform(30, 80)
                angle = self.np_random.uniform(0, 2 * math.pi)
                pos = [center[0] + math.cos(angle) * radius, center[1] + math.sin(angle) * radius]
                vel = [center, radius, angle, enemy_speed] # Special velocity format for circular
            else: # Wave 5: Homing
                pos = [self.np_random.uniform(0, self.WIDTH), self.np_random.uniform(0, 50)]
                vel = [0, 0] # Will be calculated each frame

            self.enemies.append({
                "pos": pos,
                "vel": vel,
                "type": self.current_wave,
                "fire_rate": 0.005 + wave_index * 0.004,
                "size": 10
            })
    
    def step(self, action):
        if self.auto_advance:
            self.clock.tick(30)

        if self.game_over:
            reward = 0
            terminated = True
            return self._get_observation(), reward, terminated, False, self._get_info()

        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        shift_held = action[2] == 1
        
        reward = 0.1 # Survival reward
        
        # --- Update game logic ---
        self.steps += 1
        
        # Handle inaction penalty
        is_passive = (movement == 0 and not space_held)
        if is_passive:
            self.inaction_counter += 1
        else:
            self.inaction_counter = 0
        
        if self.inaction_counter >= self.INACTION_THRESHOLD:
            reward -= 0.02
            self.inaction_counter = 0

        # Player movement & actions
        self._update_player(movement, space_held, shift_held)
        
        # Update entities
        self._update_projectiles()
        self._update_enemies()
        self._update_particles()
        
        # Collision detection
        reward += self._handle_collisions()

        # Check for wave clear
        if not self.enemies and not self.game_won:
            reward += 10
            self._start_next_wave()
            if self.game_won:
                reward += 100 # Game win reward

        # Check termination conditions
        terminated = self._check_termination()
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _update_player(self, movement, space_held, shift_held):
        # Boost
        if shift_held and self.boost_timer <= 0:
            self.boost_timer = self.BOOST_DURATION
        
        current_speed = self.PLAYER_SPEED
        if self.boost_timer > 0:
            current_speed *= self.BOOST_MULTIPLIER
            self.boost_timer -= 1
            # Add boost trail particles
            self.particles.append({
                "pos": list(self.player_pos),
                "vel": [self.np_random.uniform(-0.5, 0.5), self.np_random.uniform(1, 2)],
                "life": 10,
                "max_life": 10,
                "color": self.COLOR_PLAYER_BOOST,
                "size": 4
            })

        # Movement
        if movement == 1: self.player_pos[1] -= current_speed
        if movement == 2: self.player_pos[1] += current_speed
        if movement == 3: self.player_pos[0] -= current_speed
        if movement == 4: self.player_pos[0] += current_speed
        
        self.player_pos[0] = np.clip(self.player_pos[0], 0, self.WIDTH)
        self.player_pos[1] = np.clip(self.player_pos[1], 0, self.HEIGHT)
        
        # Firing
        if space_held and not self.last_space_held:
            # # pew pew sound
            self.projectiles.append({
                "pos": [self.player_pos[0], self.player_pos[1] - 15],
                "vel": [0, -self.PROJECTILE_SPEED],
                "owner": "player",
                "color": self.COLOR_PROJECTILE_PLAYER,
                "size": 3
            })
        self.last_space_held = space_held

    def _update_projectiles(self):
        for p in self.projectiles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            if not (0 < p["pos"][0] < self.WIDTH and 0 < p["pos"][1] < self.HEIGHT):
                self.projectiles.remove(p)

    def _update_enemies(self):
        for e in self.enemies:
            wave_type = e["type"]
            if wave_type == 1: # Horizontal
                e["pos"][0] += e["vel"][0]
                if e["pos"][0] < 0 or e["pos"][0] > self.WIDTH: e["vel"][0] *= -1
            elif wave_type == 2: # Vertical
                e["pos"][1] += e["vel"][1]
                if e["pos"][1] < 0 or e["pos"][1] > self.HEIGHT / 2: e["vel"][1] *= -1
            elif wave_type == 3: # Diagonal
                e["pos"][0] += e["vel"][0]
                e["pos"][1] += e["vel"][1]
                if e["pos"][0] < 0 or e["pos"][0] > self.WIDTH: e["vel"][0] *= -1
                if e["pos"][1] < 0 or e["pos"][1] > self.HEIGHT: e["vel"][1] *= -1
            elif wave_type == 4: # Circular
                center, radius, angle, speed = e["vel"]
                angle += speed * 0.05
                e["vel"][2] = angle
                e["pos"][0] = center[0] + math.cos(angle) * radius
                e["pos"][1] = center[1] + math.sin(angle) * radius
            elif wave_type == 5: # Homing
                direction = np.array(self.player_pos) - np.array(e["pos"])
                norm = np.linalg.norm(direction)
                if norm > 0:
                    e["vel"] = direction / norm * (1.0 + (self.current_wave - 1) * 0.5)
                e["pos"][0] += e["vel"][0]
                e["pos"][1] += e["vel"][1]

            # Enemy firing
            if self.np_random.random() < e["fire_rate"]:
                # # enemy shoot sound
                direction = np.array(self.player_pos) - np.array(e["pos"])
                norm = np.linalg.norm(direction)
                if norm > 0:
                    vel = direction / norm * (self.PROJECTILE_SPEED / 2)
                    self.projectiles.append({
                        "pos": list(e["pos"]), "vel": vel, "owner": "enemy",
                        "color": self.COLOR_PROJECTILE_ENEMY, "size": 4
                    })

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)

    def _handle_collisions(self):
        reward = 0
        # Player projectiles vs enemies
        for proj in self.projectiles[:]:
            if proj["owner"] == "player":
                for enemy in self.enemies[:]:
                    dist = math.hypot(proj["pos"][0] - enemy["pos"][0], proj["pos"][1] - enemy["pos"][1])
                    if dist < enemy["size"] + proj["size"]:
                        # # explosion sound
                        self._create_explosion(enemy["pos"], self.WAVE_COLORS[enemy["type"] - 1])
                        self.enemies.remove(enemy)
                        if proj in self.projectiles: self.projectiles.remove(proj)
                        self.score += 10
                        reward += 1
                        break
        
        # Enemy projectiles vs player
        player_size = 10
        for proj in self.projectiles[:]:
            if proj["owner"] == "enemy":
                dist = math.hypot(proj["pos"][0] - self.player_pos[0], proj["pos"][1] - self.player_pos[1])
                if dist < player_size + proj["size"]:
                    # # player hit sound
                    if proj in self.projectiles: self.projectiles.remove(proj)
                    self.player_health -= 1
                    reward -= 0.2
                    self._create_explosion(self.player_pos, (200, 200, 255), 10)
                    if self.player_health <= 0:
                        self.game_over = True
                        self._create_explosion(self.player_pos, (255, 255, 255), 50)
                    break
        return reward

    def _create_explosion(self, pos, color, count=20):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = self.np_random.integers(15, 30)
            self.particles.append({
                "pos": list(pos), "vel": vel, "life": life, "max_life": life,
                "color": color, "size": self.np_random.uniform(1, 3)
            })

    def _check_termination(self):
        if self.game_over:
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        return False
    
    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.current_wave,
            "health": self.player_health,
            "game_won": self.game_won
        }

    def _render_game(self):
        # Render stars
        for star in self.stars:
            c = int(star["color"])
            pygame.draw.circle(self.screen, (c,c,c), star["pos"], star["size"])

        # Render projectiles
        for p in self.projectiles:
            pygame.draw.circle(self.screen, p["color"], (int(p["pos"][0]), int(p["pos"][1])), p["size"])
        
        # Render enemies
        for e in self.enemies:
            color = self.WAVE_COLORS[e["type"] - 1]
            pos = (int(e["pos"][0]), int(e["pos"][1]))
            size = e["size"]
            points = [
                (pos[0], pos[1] - size),
                (pos[0] - size * 0.8, pos[1] + size * 0.6),
                (pos[0] + size * 0.8, pos[1] + size * 0.6)
            ]
            pygame.gfxdraw.aapolygon(self.screen, points, color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)

        # Render player
        if self.player_health > 0:
            color = self.HEALTH_COLORS[min(self.player_health, len(self.HEALTH_COLORS)) - 1]
            if self.boost_timer > 0:
                color = self.COLOR_PLAYER_BOOST
            
            p_pos = (int(self.player_pos[0]), int(self.player_pos[1]))
            size = 12
            points = [
                (p_pos[0], p_pos[1] - size),
                (p_pos[0] - size * 0.7, p_pos[1] + size * 0.7),
                (p_pos[0] + size * 0.7, p_pos[1] + size * 0.7)
            ]
            pygame.gfxdraw.aapolygon(self.screen, points, color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
            # Glow effect
            pygame.gfxdraw.filled_circle(self.screen, p_pos[0], p_pos[1] + 2, 5, (255, 200, 0, 150))

        # Render particles
        for p in self.particles:
            alpha = p["life"] / p["max_life"]
            color = tuple(c * alpha for c in p["color"])
            pygame.draw.circle(self.screen, color, (int(p["pos"][0]), int(p["pos"][1])), p["size"] * alpha)

    def _render_ui(self):
        # Score
        score_surf = self.font_ui.render(f"SCORE: {self.score}", True, (255, 255, 255))
        self.screen.blit(score_surf, (10, 10))
        
        # Wave
        wave_str = f"WAVE: {self.current_wave}/{self.MAX_WAVES}"
        wave_surf = self.font_ui.render(wave_str, True, (255, 255, 255))
        self.screen.blit(wave_surf, (self.WIDTH - wave_surf.get_width() - 10, 10))

        # Health
        health_color = self.HEALTH_COLORS[min(max(0, self.player_health), len(self.HEALTH_COLORS)) - 1] if self.player_health > 0 else (50,50,50)
        health_bar_width = 100
        health_bar_height = 10
        health_current_width = (self.player_health / self.PLAYER_MAX_HEALTH) * health_bar_width
        bar_x = self.WIDTH/2 - health_bar_width/2
        bar_y = self.HEIGHT - 20
        pygame.draw.rect(self.screen, (50,50,50), (bar_x, bar_y, health_bar_width, health_bar_height))
        if health_current_width > 0:
            pygame.draw.rect(self.screen, health_color, (bar_x, bar_y, health_current_width, health_bar_height))
        pygame.draw.rect(self.screen, (255,255,255), (bar_x, bar_y, health_bar_width, health_bar_height), 1)

        # Game Over / Win message
        if self.game_over:
            msg = "YOU WON!" if self.game_won else "GAME OVER"
            color = (50, 255, 50) if self.game_won else (255, 50, 50)
            msg_surf = self.font_game_over.render(msg, True, color)
            msg_rect = msg_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(msg_surf, msg_rect)

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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

# Example of how to run the environment
if __name__ == '__main__':
    # For human play
    import os
    os.environ['SDL_VIDEODRIVER'] = 'x11' # Change to 'windows' or 'mac' if needed, or remove for default
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Arcade Shooter")
    
    terminated = False
    
    print(env.user_guide)
    print(env.game_description)
    
    while not terminated:
        # --- Human Controls ---
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Rendering ---
        # The observation is already the rendered frame
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
    
    print(f"Game Over. Final Info: {info}")
    env.close()