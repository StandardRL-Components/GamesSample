import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move. Your last move direction is your aim direction. "
        "Press Space to fire. Hold Shift to cycle weapons."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Survive hordes of zombies in a top-down arena shooter. "
        "Eliminate 25 zombies to win. Use your pistol, shotgun, and rocket launcher wisely."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 10000
        self.WIN_CONDITION = 25

        # Colors
        self.COLOR_BG = (25, 25, 35)
        self.COLOR_PLAYER = (50, 255, 150)
        self.COLOR_ZOMBIE = (255, 50, 50)
        self.COLOR_WALL = (60, 60, 80)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_HEALTH_BG = (100, 0, 0)
        self.COLOR_HEALTH_FG = (0, 200, 0)
        
        # Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 72, bold=True)
        
        # Game state variables are initialized in reset()
        self.player_pos = None
        self.player_health = None
        self.player_aim_direction = None
        self.player_rect = None
        
        self.zombies = []
        self.projectiles = []
        self.explosions = []
        self.particles = []

        self.weapons = []
        self.current_weapon_idx = None
        self.weapon_cooldown = None
        self.weapon_reloading = None

        self.zombie_spawn_timer = None
        self.zombie_spawn_rate = None
        self.difficulty_timer = None
        
        self.score = None
        self.zombies_killed = None
        self.steps = None
        self.game_over = None
        self.win = None
        
        self.prev_space_held = False
        self.prev_shift_held = False

        self.rng = None
        
        self.reset()
        
        # Run self-check
        # self.validate_implementation() # Commented out for submission

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            # Fallback to a new generator if seed is None
            if self.rng is None:
                self.rng = np.random.default_rng()

        # Player state
        self.player_pos = pygame.math.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
        self.player_health = 100
        self.PLAYER_SIZE = 20
        self.PLAYER_SPEED = 4
        self.player_aim_direction = pygame.math.Vector2(1, 0)
        self.player_rect = pygame.Rect(0, 0, self.PLAYER_SIZE, self.PLAYER_SIZE)

        # Game entities
        self.zombies = []
        self.projectiles = []
        self.explosions = []
        self.particles = []

        # Weapon system
        self.weapons = [
            {"name": "Pistol", "ammo": -1, "max_ammo": -1, "cooldown": 5, "reload": 0, "damage": 10, "projectiles": 1, "spread": 0.1, "speed": 15},
            {"name": "Shotgun", "ammo": 5, "max_ammo": 5, "cooldown": 20, "reload": 45, "damage": 25, "projectiles": 6, "spread": 0.5, "speed": 12},
            {"name": "Rocket", "ammo": 2, "max_ammo": 2, "cooldown": 40, "reload": 90, "damage": 100, "projectiles": 1, "spread": 0, "speed": 8, "radius": 75}
        ]
        self.current_weapon_idx = 0
        self.weapon_cooldown = 0
        self.weapon_reloading = 0

        # Spawning logic
        self.zombie_spawn_timer = 60
        self.zombie_spawn_rate = 60
        self.difficulty_timer = 500
        
        # Game state
        self.steps = 0
        self.score = 0
        self.zombies_killed = 0
        self.game_over = False
        self.win = False

        # Input state
        self.prev_space_held = False
        self.prev_shift_held = False
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        self.steps += 1
        
        # Handle input and player actions
        reward += self._handle_input(action)
        
        # Update game world
        self._update_zombies()
        reward += self._update_projectiles()
        self._update_explosions()
        self._update_particles()
        self._update_spawners()
        
        # Update timers
        if self.weapon_cooldown > 0: self.weapon_cooldown -= 1
        if self.weapon_reloading > 0:
            self.weapon_reloading -= 1
            if self.weapon_reloading == 0:
                # sfx: reload_complete.wav
                w = self.weapons[self.current_weapon_idx]
                w["ammo"] = w["max_ammo"]

        # Check for termination conditions
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        if terminated:
            self.game_over = True
            if self.win:
                reward += 100 # Win bonus
            else:
                reward -= 100 # Loss penalty
        
        if self.auto_advance:
            self.clock.tick(self.FPS)

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Player Movement & Aiming
        move_vec = pygame.math.Vector2(0, 0)
        if movement == 1: move_vec.y = -1
        elif movement == 2: move_vec.y = 1
        elif movement == 3: move_vec.x = -1
        elif movement == 4: move_vec.x = 1
        
        if move_vec.length() > 0:
            move_vec.normalize_ip()
            self.player_pos += move_vec * self.PLAYER_SPEED
            self.player_aim_direction = move_vec
        
        self.player_pos.x = np.clip(self.player_pos.x, self.PLAYER_SIZE/2, self.WIDTH - self.PLAYER_SIZE/2)
        self.player_pos.y = np.clip(self.player_pos.y, self.PLAYER_SIZE/2, self.HEIGHT - self.PLAYER_SIZE/2)

        # Weapon Switching
        if shift_held and not self.prev_shift_held:
            # sfx: weapon_switch.wav
            self.current_weapon_idx = (self.current_weapon_idx + 1) % len(self.weapons)
            self.weapon_cooldown = 0
            self.weapon_reloading = 0

        # Firing
        reward = 0
        if space_held and not self.prev_space_held:
            reward += self._fire_weapon()

        self.prev_space_held, self.prev_shift_held = space_held, shift_held
        return reward

    def _fire_weapon(self):
        if self.weapon_cooldown > 0 or self.weapon_reloading > 0:
            return 0
        
        weapon = self.weapons[self.current_weapon_idx]
        
        if weapon["ammo"] == 0:
            self.weapon_reloading = weapon["reload"]
            # sfx: empty_clip.wav
            return 0

        # sfx: depending on weapon (pistol_shot.wav, shotgun_blast.wav, rocket_launch.wav)
        self.weapon_cooldown = weapon["cooldown"]
        if weapon["ammo"] > 0:
            weapon["ammo"] -= 1

        # Create muzzle flash
        flash_pos = self.player_pos + self.player_aim_direction * (self.PLAYER_SIZE / 2 + 5)
        for _ in range(20):
            speed = self.rng.uniform(2, 6)
            angle = self.player_aim_direction.angle_to(pygame.math.Vector2(1,0)) + self.rng.uniform(-45, 45)
            vel = pygame.math.Vector2(speed, 0).rotate(-angle)
            self.particles.append({
                "pos": pygame.math.Vector2(flash_pos), "vel": vel, "lifespan": self.rng.integers(5, 10),
                "color": random.choice([(255, 255, 0), (255, 200, 0), (255, 255, 255)])
            })

        # Create projectiles
        for _ in range(weapon["projectiles"]):
            angle_offset = self.rng.uniform(-weapon["spread"], weapon["spread"]) * 90
            vel = self.player_aim_direction.rotate(angle_offset) * weapon["speed"]
            self.projectiles.append({
                "pos": pygame.math.Vector2(self.player_pos), "vel": vel, "damage": weapon["damage"],
                "type": weapon["name"]
            })
        
        if weapon["ammo"] == 0 and weapon["reload"] > 0:
            self.weapon_reloading = weapon["reload"]
            # sfx: start_reload.wav
            
        return -0.01 # Small penalty for firing

    def _update_zombies(self):
        for z in self.zombies:
            direction = (self.player_pos - z["pos"]).normalize()
            z["pos"] += direction * z["speed"]
            z["rect"].center = z["pos"]
            
            # Zombie-Player collision
            if self.player_rect.colliderect(z["rect"]):
                self.player_health -= 1
                # sfx: player_hurt.wav
                self._create_damage_vignette()
                z["pos"] -= direction * (z["size"] / 2) # Push back

    def _update_projectiles(self):
        reward = 0
        projectiles_to_remove = []
        for i, p in enumerate(self.projectiles):
            p["pos"] += p["vel"]
            
            # Out of bounds check
            if not (0 < p["pos"].x < self.WIDTH and 0 < p["pos"].y < self.HEIGHT):
                if i not in projectiles_to_remove:
                    projectiles_to_remove.append(i)
                if p["type"] == "Rocket":
                    self._create_explosion(p["pos"], self.weapons[2]["radius"], p["damage"])
                continue

            # Zombie collision check
            hit = False
            for z_idx, z in enumerate(self.zombies):
                if z["rect"].collidepoint(p["pos"]):
                    # sfx: zombie_hit.wav
                    z["health"] -= p["damage"]
                    reward += 0.1 # Hit reward
                    self._create_blood_splatter(p["pos"], p["vel"])
                    hit = True
                    if z["health"] <= 0:
                        # sfx: zombie_die.wav
                        self.zombies.pop(z_idx)
                        self.score += 1
                        self.zombies_killed += 1
                        reward += 1 # Kill reward
                    break 
            
            if hit:
                if i not in projectiles_to_remove:
                    projectiles_to_remove.append(i)
                if p["type"] == "Rocket":
                     self._create_explosion(p["pos"], self.weapons[2]["radius"], p["damage"])

        # Remove projectiles in reverse order
        for i in sorted(list(set(projectiles_to_remove)), reverse=True):
            self.projectiles.pop(i)
            
        return reward

    def _create_explosion(self, pos, radius, damage):
        self.explosions.append({"pos": pos, "radius": radius, "damage": damage, "lifespan": 5})
        # sfx: explosion.wav
        # Visual effect
        for _ in range(100):
            speed = self.rng.uniform(1, 10)
            angle = self.rng.uniform(0, 360)
            vel = pygame.math.Vector2(speed, 0).rotate(angle)
            self.particles.append({
                "pos": pygame.math.Vector2(pos), "vel": vel, "lifespan": self.rng.integers(15, 30),
                "color": random.choice([(255, 100, 0), (255, 200, 0), (150, 150, 150)])
            })

    def _update_explosions(self):
        explosions_to_remove = []
        for i, exp in enumerate(self.explosions):
            if exp["lifespan"] == 5: # Damage on first frame only
                zombies_to_remove = []
                for z_idx, z in enumerate(self.zombies):
                    if z["pos"].distance_to(exp["pos"]) < exp["radius"]:
                        z["health"] -= exp["damage"]
                        self._create_blood_splatter(z["pos"], (z["pos"]-exp["pos"]).normalize())
                        if z["health"] <= 0:
                            if z_idx not in zombies_to_remove:
                                zombies_to_remove.append(z_idx)
                
                for z_idx in sorted(zombies_to_remove, reverse=True):
                    self.zombies.pop(z_idx)
                    self.score += 1
                    self.zombies_killed += 1
                    # No extra reward here to avoid double counting with projectile hit
            
            exp["lifespan"] -= 1
            if exp["lifespan"] <= 0:
                explosions_to_remove.append(i)
        
        for i in sorted(explosions_to_remove, reverse=True):
            self.explosions.pop(i)

    def _update_particles(self):
        self.particles = [p for p in self.particles if p["lifespan"] > 0]
        for p in self.particles:
            p["pos"] += p["vel"]
            p["lifespan"] -= 1
            p["vel"] *= 0.95 # Damping

    def _update_spawners(self):
        # Increase difficulty over time
        self.difficulty_timer -= 1
        if self.difficulty_timer <= 0:
            self.zombie_spawn_rate = max(15, self.zombie_spawn_rate - 5)
            self.difficulty_timer = 500

        # Spawn zombies
        self.zombie_spawn_timer -= 1
        if self.zombie_spawn_timer <= 0:
            self.zombie_spawn_timer = self.zombie_spawn_rate
            
            side = self.rng.integers(0, 4)
            if side == 0: # top
                pos = pygame.math.Vector2(self.rng.uniform(0, self.WIDTH), -20)
            elif side == 1: # bottom
                pos = pygame.math.Vector2(self.rng.uniform(0, self.WIDTH), self.HEIGHT + 20)
            elif side == 2: # left
                pos = pygame.math.Vector2(-20, self.rng.uniform(0, self.HEIGHT))
            else: # right
                pos = pygame.math.Vector2(self.WIDTH + 20, self.rng.uniform(0, self.HEIGHT))
            
            size = self.rng.integers(12, 18)
            speed = self.rng.uniform(0.8, 1.5)
            health = 100 + self.rng.integers(-10, 20) * (self.steps / 1000)
            self.zombies.append({
                "pos": pos, "size": size, "speed": speed, "health": health,
                "rect": pygame.Rect(0, 0, size, size)
            })

    def _create_blood_splatter(self, pos, direction):
        for _ in range(15):
            speed = self.rng.uniform(1, 4)
            angle = direction.angle_to(pygame.math.Vector2(1,0)) + self.rng.uniform(-60, 60)
            vel = pygame.math.Vector2(speed, 0).rotate(-angle)
            self.particles.append({
                "pos": pygame.math.Vector2(pos), "vel": vel, "lifespan": self.rng.integers(10, 20),
                "color": (180, 0, 0)
            })

    def _create_damage_vignette(self):
         for _ in range(30):
            pos = pygame.math.Vector2(self.rng.choice([0, self.WIDTH]), self.rng.uniform(0, self.HEIGHT))
            if self.rng.choice([True, False]):
                pos = pygame.math.Vector2(self.rng.uniform(0, self.WIDTH), self.rng.choice([0, self.HEIGHT]))
            
            target = pygame.math.Vector2(self.WIDTH/2, self.HEIGHT/2)
            vel = (target - pos).normalize() * self.rng.uniform(1, 5)
            self.particles.append({
                "pos": pos, "vel": vel, "lifespan": self.rng.integers(15, 25),
                "color": (150, 0, 0)
            })
            
    def _check_termination(self):
        if self.player_health <= 0:
            self.win = False
            return True
        if self.zombies_killed >= self.WIN_CONDITION:
            self.win = True
            return True
        return False
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw explosions first (underneath everything)
        for exp in self.explosions:
            r = exp["radius"] * (1.0 - exp["lifespan"] / 5.0)
            alpha = int(255 * (exp["lifespan"] / 5.0))
            # Create a temporary surface for the circle to handle alpha
            temp_surf = pygame.Surface((int(r*2), int(r*2)), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(temp_surf, int(r), int(r), int(r), (255, 150, 0, alpha))
            self.screen.blit(temp_surf, (int(exp["pos"].x - r), int(exp["pos"].y - r)))

        # Draw particles
        for p in self.particles:
            pygame.draw.circle(self.screen, p["color"], p["pos"], int(p["lifespan"] / 5 + 1))

        # Draw zombies
        for z in self.zombies:
            pygame.draw.rect(self.screen, self.COLOR_ZOMBIE, z["rect"])
            
        # Draw player
        self.player_rect.center = self.player_pos
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, self.player_rect, border_radius=3)
        # Draw aim indicator
        aim_end = self.player_pos + self.player_aim_direction * (self.PLAYER_SIZE * 0.75)
        pygame.draw.line(self.screen, (255, 255, 255), self.player_pos, aim_end, 2)

        # Draw projectiles
        for p in self.projectiles:
            if p["type"] == "Rocket":
                pygame.draw.circle(self.screen, (255, 150, 0), p["pos"], 5)
                pygame.gfxdraw.aacircle(self.screen, int(p["pos"].x), int(p["pos"].y), 5, (255, 255, 0))
            else:
                pygame.draw.circle(self.screen, (255, 255, 100), p["pos"], 3)

    def _render_ui(self):
        # Health Bar
        health_pct = max(0, self.player_health / 100)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BG, (10, 10, 200, 20))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_FG, (10, 10, 200 * health_pct, 20))
        health_text = self.font_small.render(f"{int(self.player_health)}/100", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (15, 12))

        # Score
        score_text = self.font_small.render(f"KILLS: {self.zombies_killed}/{self.WIN_CONDITION}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 10))
        
        # Weapon/Ammo
        w = self.weapons[self.current_weapon_idx]
        ammo_str = "INF" if w['ammo'] < 0 else f"{w['ammo']}/{w['max_ammo']}"
        if self.weapon_reloading > 0:
            ammo_str = f"RELOADING... {self.weapon_reloading/self.FPS:.1f}s"
        weapon_text = self.font_small.render(f"{w['name']}: {ammo_str}", True, self.COLOR_TEXT)
        self.screen.blit(weapon_text, (10, 35))
        
        # Game Over Screen
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            message = "VICTORY" if self.win else "YOU DIED"
            color = (50, 255, 50) if self.win else (255, 50, 50)
            end_text = self.font_large.render(message, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "zombies_killed": self.zombies_killed,
            "player_health": self.player_health,
            "current_weapon": self.weapons[self.current_weapon_idx]["name"],
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
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    # Re-enable the display driver for human play
    os.environ["SDL_VIDEODRIVER"] = "x11"
    
    env = GameEnv()
    obs, info = env.reset()
    
    # Setup a window to display the game
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Zombie Arena")
    clock = pygame.time.Clock()
    
    done = False
    while not done:
        # --- Human Controls ---
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Step the environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # --- Render the observation to the window ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event handling (for closing the window) ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        clock.tick(env.FPS)

        if done:
            print(f"Game Over! Final Info: {info}")
            # Wait a bit before closing
            pygame.time.wait(3000)

    env.close()