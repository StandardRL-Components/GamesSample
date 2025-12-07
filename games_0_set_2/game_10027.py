import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:49:25.728144
# Source Brief: brief_00027.md
# Brief Index: 27
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment for a retro action-platformer.
    The player controls a magnetic robo-warrior, navigating levels,
    fighting enemies, and using a camouflage ability.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Control a magnetic robo-warrior in a retro action-platformer. "
        "Navigate levels, fight enemies, and use a camouflage ability to survive."
    )
    user_guide = (
        "Controls: ←→ to move, ↑ to jump. Press space to fire and hold shift to activate camouflage."
    )
    auto_advance = True

    # --- CONSTANTS ---
    # Screen
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400

    # Colors
    COLOR_BG = (15, 15, 25)
    COLOR_BG_CIRCUIT = (25, 25, 45)
    COLOR_PLATFORM = (60, 60, 80)
    COLOR_PLAYER = (50, 255, 150)
    COLOR_PLAYER_GLOW = (50, 255, 150, 50)
    COLOR_PLAYER_CAMO = (50, 150, 255, 150)
    COLOR_ENEMY = (255, 80, 80)
    COLOR_ENEMY_GLOW = (255, 80, 80, 50)
    COLOR_PROJECTILE = (100, 200, 255)
    COLOR_EXIT = (255, 220, 0)
    COLOR_TEXT = (220, 220, 240)
    COLOR_HEALTH_BAR = (50, 200, 50)
    COLOR_CAMO_BAR = (50, 150, 255)
    COLOR_BAR_BG = (40, 40, 60)

    # Physics & Gameplay
    FPS = 30
    GRAVITY = 0.6
    PLAYER_JUMP_STRENGTH = -11
    PLAYER_ACCEL = 1.2
    PLAYER_FRICTION = 0.85
    PLAYER_MAX_SPEED = 7
    MAX_EPISODE_STEPS = 2000
    PROJECTILE_SPEED = 12
    FIRE_COOLDOWN = 6  # frames

    # Player Stats
    PLAYER_MAX_HEALTH = 100
    PLAYER_MAX_CAMO_ENERGY = 100
    CAMO_DRAIN_RATE = 1.5
    CAMO_RECHARGE_RATE = 0.5

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        self.render_mode = render_mode

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_level = pygame.font.SysFont("monospace", 24, bold=True)

        # Game state variables are initialized in reset()
        self.player = {}
        self.platforms = []
        self.enemies = []
        self.projectiles = []
        self.particles = []
        self.exit_portal = None
        
        self.steps = 0
        self.score = 0
        self.level = 1
        self.game_over = False
        self.last_dist_to_exit = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        
        if options and 'level' in options:
            self.level = options.get('level', 1)
        # On a full reset (not just level change), reset level to 1
        elif not (options and options.get('keep_level', False)):
             self.level = 1

        self._generate_level()

        self.player = {
            "pos": pygame.Vector2(self.spawn_pos.x, self.spawn_pos.y),
            "vel": pygame.Vector2(0, 0),
            "radius": 12,
            "health": self.PLAYER_MAX_HEALTH,
            "camo_energy": self.PLAYER_MAX_CAMO_ENERGY,
            "is_camouflaged": False,
            "on_ground": False,
            "jumps_left": 1,
            "last_move_dir": pygame.Vector2(1, 0),
            "fire_cooldown": 0,
            "invulnerable_timer": 0
        }
        
        self.projectiles.clear()
        self.particles.clear()
        
        if self.exit_portal:
            self.last_dist_to_exit = self.player["pos"].distance_to(self.exit_portal.center)

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        self.steps += 1
        reward = 0.0

        # --- UPDATE STATE ---
        self._update_player(movement, space_held, shift_held)
        self._update_enemies()
        self._update_projectiles()
        self._update_particles()
        
        # --- CALCULATE REWARDS ---
        # Distance to exit reward
        dist_to_exit = self.player["pos"].distance_to(self.exit_portal.center)
        reward += (self.last_dist_to_exit - dist_to_exit) * 0.1
        self.last_dist_to_exit = dist_to_exit
        
        # Camouflage reward
        if self.player["is_camouflaged"]:
            for enemy in self.enemies:
                if self.player["pos"].distance_to(enemy["pos"]) < enemy["detect_radius"]:
                    reward += 0.01

        # --- HANDLE COLLISIONS ---
        reward += self._handle_collisions()

        self.score += reward
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_EPISODE_STEPS
        if truncated:
            self.game_over = True

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _generate_level(self):
        self.platforms = []
        self.enemies = []
        
        # Static floor
        self.platforms.append(pygame.Rect(0, self.SCREEN_HEIGHT - 20, self.SCREEN_WIDTH, 20))
        
        level_layout = (self.level - 1) % 3
        enemy_speed = 1.0 + (self.level // 5) * 0.2

        if level_layout == 0:
            self.spawn_pos = pygame.Vector2(50, 350)
            self.exit_portal = pygame.Rect(self.SCREEN_WIDTH - 80, 50, 40, 40)
            self.platforms.extend([
                pygame.Rect(150, 300, 150, 20),
                pygame.Rect(350, 220, 150, 20),
                pygame.Rect(150, 140, 150, 20),
                pygame.Rect(self.SCREEN_WIDTH - 190, 90, 150, 20)
            ])
            self._spawn_enemy(pygame.Rect(150, 300, 150, 20), speed=enemy_speed)
            self._spawn_enemy(pygame.Rect(350, 220, 150, 20), speed=enemy_speed)

        elif level_layout == 1:
            self.spawn_pos = pygame.Vector2(50, 100)
            self.exit_portal = pygame.Rect(self.SCREEN_WIDTH - 80, 330, 40, 40)
            self.platforms.extend([
                pygame.Rect(0, 150, 150, 20),
                pygame.Rect(250, 250, 100, 20),
                pygame.Rect(50, 320, 150, 20),
                pygame.Rect(400, 150, 100, 20),
                pygame.Rect(self.SCREEN_WIDTH - 150, 370, 150, 20)
            ])
            self._spawn_enemy(pygame.Rect(0, 150, 150, 20), speed=enemy_speed)
            self._spawn_enemy(pygame.Rect(self.SCREEN_WIDTH - 150, 370, 150, 20), speed=enemy_speed)

        elif level_layout == 2:
            self.spawn_pos = pygame.Vector2(320, 50)
            self.exit_portal = pygame.Rect(300, 330, 40, 40)
            self.platforms.extend([
                pygame.Rect(280, 90, 80, 20),
                pygame.Rect(100, 180, 100, 20),
                pygame.Rect(440, 180, 100, 20),
                pygame.Rect(0, 260, self.SCREEN_WIDTH, 20),
                pygame.Rect(250, 370, 140, 20)
            ])
            self._spawn_enemy(pygame.Rect(0, 260, self.SCREEN_WIDTH, 20), speed=enemy_speed, patrol_range=0.3)
            self._spawn_enemy(pygame.Rect(0, 260, self.SCREEN_WIDTH, 20), speed=enemy_speed, patrol_range=0.8)

    def _spawn_enemy(self, platform, speed=1.0, patrol_range=0.9):
        patrol_width = platform.width * patrol_range
        start_x = platform.centerx - patrol_width / 2
        end_x = platform.centerx + patrol_width / 2
        self.enemies.append({
            "pos": pygame.Vector2(start_x, platform.top - 10),
            "radius": 10,
            "speed": speed,
            "direction": 1,
            "patrol_bounds": (start_x, end_x),
            "detect_radius": 120
        })

    def _update_player(self, movement, space_held, shift_held):
        # Handle camouflage
        if shift_held and self.player["camo_energy"] > 0:
            self.player["is_camouflaged"] = True
            self.player["camo_energy"] = max(0, self.player["camo_energy"] - self.CAMO_DRAIN_RATE)
        else:
            self.player["is_camouflaged"] = False
            self.player["camo_energy"] = min(self.PLAYER_MAX_CAMO_ENERGY, self.player["camo_energy"] + self.CAMO_RECHARGE_RATE)
        
        # Handle movement
        if movement == 1 and self.player["on_ground"]:  # Up / Jump
            # Player upgrade at level 5
            max_jumps = 2 if self.level >= 5 else 1
            if self.player["jumps_left"] < max_jumps:
                self.player["vel"].y = self.PLAYER_JUMP_STRENGTH
                self.player["jumps_left"] += 1
                # sfx: jump
        if movement == 3:  # Left
            self.player["vel"].x -= self.PLAYER_ACCEL
            self.player["last_move_dir"] = pygame.Vector2(-1, 0)
        if movement == 4:  # Right
            self.player["vel"].x += self.PLAYER_ACCEL
            self.player["last_move_dir"] = pygame.Vector2(1, 0)

        # Handle shooting
        if self.player["fire_cooldown"] > 0:
            self.player["fire_cooldown"] -= 1
        if space_held and self.player["fire_cooldown"] == 0:
            self._fire_projectile()
            self.player["fire_cooldown"] = self.FIRE_COOLDOWN

        # Apply physics
        self.player["vel"].y += self.GRAVITY
        self.player["vel"].x *= self.PLAYER_FRICTION
        self.player["vel"].x = max(-self.PLAYER_MAX_SPEED, min(self.PLAYER_MAX_SPEED, self.player["vel"].x))

        # Update position
        self.player["pos"].x += self.player["vel"].x
        self._handle_platform_collisions_x()
        self.player["pos"].y += self.player["vel"].y
        self._handle_platform_collisions_y()

        # Update invulnerability
        if self.player["invulnerable_timer"] > 0:
            self.player["invulnerable_timer"] -= 1
            
        # Keep player in bounds
        self.player["pos"].x = max(self.player["radius"], min(self.SCREEN_WIDTH - self.player["radius"], self.player["pos"].x))
        if self.player["pos"].y > self.SCREEN_HEIGHT + 50: # Fell off map
            self.player["health"] = 0

    def _fire_projectile(self):
        # sfx: shoot
        start_pos = pygame.Vector2(self.player["pos"])
        velocity = self.player["last_move_dir"] * self.PROJECTILE_SPEED
        # Add slight upward angle for better feel
        velocity.y -= 2
        self.projectiles.append({
            "pos": start_pos,
            "vel": velocity,
            "radius": 5,
            "trail": []
        })

    def _handle_platform_collisions_x(self):
        player_rect = pygame.Rect(self.player["pos"].x - self.player["radius"], self.player["pos"].y - self.player["radius"], self.player["radius"]*2, self.player["radius"]*2)
        for platform in self.platforms:
            if player_rect.colliderect(platform):
                if self.player["vel"].x > 0:
                    player_rect.right = platform.left
                    self.player["vel"].x = 0
                elif self.player["vel"].x < 0:
                    player_rect.left = platform.right
                    self.player["vel"].x = 0
                self.player["pos"].x = player_rect.centerx

    def _handle_platform_collisions_y(self):
        self.player["on_ground"] = False
        player_rect = pygame.Rect(self.player["pos"].x - self.player["radius"], self.player["pos"].y - self.player["radius"], self.player["radius"]*2, self.player["radius"]*2)
        for platform in self.platforms:
            if player_rect.colliderect(platform):
                if self.player["vel"].y > 0:
                    player_rect.bottom = platform.top
                    self.player["vel"].y = 0
                    self.player["on_ground"] = True
                    self.player["jumps_left"] = 0
                elif self.player["vel"].y < 0:
                    player_rect.top = platform.bottom
                    self.player["vel"].y = 0
                self.player["pos"].y = player_rect.centery
    
    def _update_enemies(self):
        for enemy in self.enemies:
            enemy["pos"].x += enemy["speed"] * enemy["direction"]
            if enemy["pos"].x <= enemy["patrol_bounds"][0] or enemy["pos"].x >= enemy["patrol_bounds"][1]:
                enemy["direction"] *= -1

    def _update_projectiles(self):
        for proj in self.projectiles[:]:
            proj["trail"].append(pygame.Vector2(proj["pos"]))
            if len(proj["trail"]) > 5:
                proj["trail"].pop(0)

            proj["vel"].y += self.GRAVITY * 0.2 # Projectile gravity
            proj["pos"] += proj["vel"]
            if not (0 < proj["pos"].x < self.SCREEN_WIDTH and 0 < proj["pos"].y < self.SCREEN_HEIGHT):
                self.projectiles.remove(proj)

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["lifespan"] -= 1
            if p["lifespan"] <= 0:
                self.particles.remove(p)

    def _create_explosion(self, pos, color, num_particles=30, max_speed=5, lifespan=20):
        # sfx: explosion
        for _ in range(num_particles):
            self.particles.append({
                "pos": pygame.Vector2(pos),
                "vel": pygame.Vector2(random.uniform(-max_speed, max_speed), random.uniform(-max_speed, max_speed)),
                "lifespan": random.randint(lifespan // 2, lifespan),
                "color": color,
                "radius": random.uniform(1, 4)
            })

    def _handle_collisions(self):
        reward = 0
        
        # Projectiles vs enemies/platforms
        for proj in self.projectiles[:]:
            # vs platforms
            proj_rect = pygame.Rect(proj["pos"].x-proj["radius"], proj["pos"].y-proj["radius"], proj["radius"]*2, proj["radius"]*2)
            if proj_rect.collidelist(self.platforms) != -1:
                self._create_explosion(proj["pos"], self.COLOR_PROJECTILE, 10, 3, 15)
                self.projectiles.remove(proj)
                continue
            
            # vs enemies
            for enemy in self.enemies[:]:
                if proj["pos"].distance_to(enemy["pos"]) < proj["radius"] + enemy["radius"]:
                    self._create_explosion(enemy["pos"], self.COLOR_ENEMY, 40, 6, 25)
                    self.enemies.remove(enemy)
                    if proj in self.projectiles:
                        self.projectiles.remove(proj)
                    reward += 5
                    break

        # Player vs enemies
        if self.player["invulnerable_timer"] == 0:
            player_rect = pygame.Rect(self.player["pos"].x-self.player["radius"], self.player["pos"].y-self.player["radius"], self.player["radius"]*2, self.player["radius"]*2)
            is_detected = not self.player["is_camouflaged"]
            for enemy in self.enemies:
                if is_detected and player_rect.colliderect(pygame.Rect(enemy["pos"].x-enemy["radius"], enemy["pos"].y-enemy["radius"], enemy["radius"]*2, enemy["radius"]*2)):
                    damage = 25
                    self.player["health"] -= damage
                    reward -= 0.5 * damage
                    self.player["invulnerable_timer"] = 60 # 2 seconds of invulnerability
                    self.player["vel"].y = -5 # Knockback
                    self._create_explosion(self.player["pos"], self.COLOR_PLAYER, 15, 4, 15)
                    # sfx: player_hit
                    break
        
        # Player vs exit portal
        player_rect = pygame.Rect(self.player["pos"].x-self.player["radius"], self.player["pos"].y-self.player["radius"], self.player["radius"]*2, self.player["radius"]*2)
        if player_rect.colliderect(self.exit_portal):
            reward += 100
            self.score += 100 # Add terminal reward to score
            self.level += 1
            # Reset for next level
            self.reset(options={'keep_level': True})

        return reward

    def _check_termination(self):
        if self.player["health"] <= 0:
            self.game_over = True
            self.score -= 100
            self._create_explosion(self.player["pos"], self.COLOR_PLAYER, 100, 8, 40)
            return True
        return False

    def _get_observation(self):
        self._render_all()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level,
            "health": self.player.get("health", 0),
            "camo_energy": self.player.get("camo_energy", 0),
        }

    def _render_all(self):
        self._render_background()
        self._render_game()
        self._render_ui()

    def _render_background(self):
        self.screen.fill(self.COLOR_BG)
        for i in range(0, self.SCREEN_WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_BG_CIRCUIT, (i, 0), (i, self.SCREEN_HEIGHT))
        for i in range(0, self.SCREEN_HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_BG_CIRCUIT, (0, i), (self.SCREEN_WIDTH, i))

    def _render_game(self):
        # Platforms
        for platform in self.platforms:
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM, platform)

        # Exit Portal
        pulse = (math.sin(self.steps * 0.1) + 1) / 2
        color = self.COLOR_EXIT
        pygame.draw.rect(self.screen, color, self.exit_portal)
        pygame.gfxdraw.rectangle(self.screen, self.exit_portal, tuple(int(c * (0.5 + pulse * 0.5)) for c in color))

        # Projectiles
        for proj in self.projectiles:
            # Trail
            if proj["trail"]:
                for i, p in enumerate(proj["trail"]):
                    alpha = int(255 * (i / len(proj["trail"])))
                    pygame.gfxdraw.filled_circle(self.screen, int(p.x), int(p.y), int(proj["radius"] * (i/len(proj["trail"]))), (*self.COLOR_PROJECTILE, alpha))
            # Main projectile
            pygame.gfxdraw.filled_circle(self.screen, int(proj["pos"].x), int(proj["pos"].y), proj["radius"], self.COLOR_PROJECTILE)

        # Enemies
        for enemy in self.enemies:
            pos = (int(enemy["pos"].x), int(enemy["pos"].y))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], enemy["radius"] + 5, self.COLOR_ENEMY_GLOW)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], enemy["radius"], self.COLOR_ENEMY)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], enemy["radius"], self.COLOR_ENEMY)

        # Player
        if self.player.get("health", 0) > 0:
            pos = (int(self.player["pos"].x), int(self.player["pos"].y))
            
            # Invulnerability flash
            if self.player["invulnerable_timer"] > 0 and self.steps % 4 < 2:
                pass # Don't draw player to make it flash
            else:
                color = self.COLOR_PLAYER_CAMO if self.player["is_camouflaged"] else self.COLOR_PLAYER
                glow_color = (*color[:3], 50) if not self.player["is_camouflaged"] else (*color[:3], 100)
                
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.player["radius"] + 8, glow_color)
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.player["radius"], color)
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.player["radius"], color)

        # Particles
        for p in self.particles:
            alpha = int(255 * (p["lifespan"] / 20.0))
            alpha = max(0, min(255, alpha))
            color = (*p["color"][:3], alpha)
            pygame.gfxdraw.filled_circle(self.screen, int(p["pos"].x), int(p["pos"].y), int(p["radius"]), color)
            
    def _render_ui(self):
        # Health Bar
        health_pct = self.player.get("health", 0) / self.PLAYER_MAX_HEALTH
        bar_w = 150
        pygame.draw.rect(self.screen, self.COLOR_BAR_BG, (10, 10, bar_w, 20))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (10, 10, int(bar_w * health_pct), 20))
        
        # Camo Bar
        camo_pct = self.player.get("camo_energy", 0) / self.PLAYER_MAX_CAMO_ENERGY
        pygame.draw.rect(self.screen, self.COLOR_BAR_BG, (10, 35, bar_w, 10))
        pygame.draw.rect(self.screen, self.COLOR_CAMO_BAR, (10, 35, int(bar_w * camo_pct), 10))

        # Score
        score_text = self.font_ui.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 10, 10))
        
        # Level
        level_text = self.font_level.render(f"LEVEL {self.level}", True, self.COLOR_TEXT)
        self.screen.blit(level_text, (self.SCREEN_WIDTH // 2 - level_text.get_width() // 2, 10))

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

if __name__ == '__main__':
    # Example of how to run the environment
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play ---
    # Use arrow keys for movement, space to shoot, shift to camouflage
    obs, info = env.reset()
    done = False
    
    # Pygame window for human interaction
    pygame.display.init()
    render_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Robo-Warrior Gym Environment")
    clock = pygame.time.Clock()
    
    total_reward = 0
    
    while not done:
        # Map keyboard inputs to the action space
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP]: movement = 1
        # K_DOWN is not used for jumping in platformers
        if keys[pygame.K_LEFT]: movement = 3
        if keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        done = terminated or truncated
        
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        clock.tick(GameEnv.FPS)

    print(f"Episode finished. Total reward: {total_reward}, Score: {info['score']}, Steps: {info['steps']}")
    env.close()