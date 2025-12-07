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

    # User-facing control string
    user_guide = (
        "Controls: ↑↓←→ to move. Hold shift to rotate aiming direction. Press space to fire."
    )

    # User-facing game description
    game_description = (
        "Pilot a robot in a top-down arena, blasting enemies to achieve ultimate robotic dominance."
    )

    # Frames auto-advance for real-time gameplay
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and world dimensions
        self.WIDTH, self.HEIGHT = 640, 400

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 50)

        # Colors
        self.COLOR_BG = (15, 20, 30)
        self.COLOR_ARENA = (40, 50, 60)
        self.COLOR_PLAYER = (0, 150, 255)
        self.COLOR_PLAYER_GLOW = (0, 150, 255, 50)
        self.COLOR_ENEMY = (255, 50, 50)
        self.COLOR_ENEMY_FLASH = (255, 255, 255)
        self.COLOR_PROJECTILE = (255, 200, 0)
        self.COLOR_HEALTH = (0, 200, 100)
        self.COLOR_HEALTH_BG = (100, 0, 0)
        self.COLOR_TEXT = (220, 220, 220)
        self.PARTICLE_COLORS = [(255, 100, 0), (255, 165, 0), (255, 200, 0)]

        # Game constants
        self.MAX_STEPS = 1500  # Increased for better gameplay
        self.PLAYER_SPEED = 3.0
        self.PLAYER_SIZE = 16
        self.PLAYER_MAX_HEALTH = 30
        self.PROJECTILE_SPEED = 8.0
        self.PROJECTILE_SIZE = 4
        self.PROJECTILE_COOLDOWN = 6  # frames
        self.ENEMY_SIZE = 14
        self.ENEMY_BASE_SPEED = 0.75
        self.ENEMY_MAX_HEALTH = 5
        self.TOTAL_ENEMIES = 20

        # Pre-calculate enemy spawn points
        self.spawn_points = self._generate_spawn_points()

        # Initialize state variables
        self.player_pos = None
        self.player_health = 0
        self.player_aim_angle = 0.0
        self.enemies = []
        self.projectiles = []
        self.particles = []
        self.score = 0
        self.steps = 0
        self.kills = 0
        self.game_over = False
        self.last_space_held = False
        self.last_shift_held = False
        self.projectile_timer = 0

        self.reset()
        
        # This check is disabled as it's not part of the final required code.
        # self.validate_implementation()

    def _generate_spawn_points(self):
        points = []
        margin = 50
        for _ in range(self.TOTAL_ENEMIES):
            side = random.randint(0, 3)
            if side == 0: # top
                points.append(pygame.Vector2(random.randint(margin, self.WIDTH - margin), margin))
            elif side == 1: # bottom
                points.append(pygame.Vector2(random.randint(margin, self.WIDTH - margin), self.HEIGHT - margin))
            elif side == 2: # left
                points.append(pygame.Vector2(margin, random.randint(margin, self.HEIGHT - margin)))
            else: # right
                points.append(pygame.Vector2(self.WIDTH - margin, random.randint(margin, self.HEIGHT - margin)))
        return points

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.player_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
        self.player_health = self.PLAYER_MAX_HEALTH
        self.player_aim_angle = 0.0
        
        self.enemies = []
        for i in range(self.TOTAL_ENEMIES):
            self.enemies.append({
                "pos": self.spawn_points[i].copy(),
                "health": self.ENEMY_MAX_HEALTH,
                "flash_timer": 0
            })

        self.projectiles = []
        self.particles = []
        self.score = 0
        self.steps = 0
        self.kills = 0
        self.game_over = False
        self.last_space_held = False
        self.last_shift_held = False
        self.projectile_timer = 0
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = -0.01  # Small penalty for time passing

        self._handle_input(action)
        self._update_player()
        reward += self._update_enemies()
        reward += self._update_projectiles()
        self._update_particles()
        
        self.steps += 1
        terminated = self._check_termination()

        if terminated and self.player_health > 0 and len(self.enemies) == 0:
            reward += 100.0 # Victory bonus

        self.score = max(0, self.score + reward) # Score can't be negative
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # Movement
        move_vec = pygame.Vector2(0, 0)
        if movement == 1: move_vec.y = -1
        elif movement == 2: move_vec.y = 1
        elif movement == 3: move_vec.x = -1
        elif movement == 4: move_vec.x = 1
        
        if move_vec.length() > 0:
            move_vec.normalize_ip()
            self.player_pos += move_vec * self.PLAYER_SPEED

        # Rotation (on press)
        shift_pressed = shift_held and not self.last_shift_held
        if shift_pressed:
            self.player_aim_angle = (self.player_aim_angle + math.pi / 4) % (2 * math.pi)

        # Shooting (on press with cooldown)
        space_pressed = space_held and not self.last_space_held
        if space_pressed and self.projectile_timer <= 0:
            # sfx: player_shoot.wav
            vel = pygame.Vector2(math.cos(self.player_aim_angle), math.sin(self.player_aim_angle)) * self.PROJECTILE_SPEED
            self.projectiles.append({"pos": self.player_pos.copy(), "vel": vel})
            self.projectile_timer = self.PROJECTILE_COOLDOWN
            # Muzzle flash
            for _ in range(5):
                self.particles.append(self._create_particle(self.player_pos, self.COLOR_PROJECTILE, 2, 5, self.player_aim_angle, spread=0.5))

        self.last_space_held = space_held
        self.last_shift_held = shift_held
        if self.projectile_timer > 0:
            self.projectile_timer -= 1

    def _update_player(self):
        # Clamp player position to arena boundaries
        arena_margin = 10
        self.player_pos.x = np.clip(self.player_pos.x, arena_margin, self.WIDTH - arena_margin)
        self.player_pos.y = np.clip(self.player_pos.y, arena_margin, self.HEIGHT - arena_margin)

    def _update_enemies(self):
        reward = 0
        current_enemy_speed = self.ENEMY_BASE_SPEED + (self.kills // 5) * 0.05

        for enemy in self.enemies:
            # Move towards player
            direction = self.player_pos - enemy["pos"]
            if direction.length() > 0:
                direction.normalize_ip()
                enemy["pos"] += direction * current_enemy_speed

            # Check collision with player
            if enemy["pos"].distance_to(self.player_pos) < self.PLAYER_SIZE:
                # sfx: player_hit.wav
                self.player_health -= 1
                reward -= 1.0 # Penalty for getting hit
                # Knockback enemy to prevent instant multi-hits
                enemy["pos"] -= direction * (self.PLAYER_SIZE * 2)

            # Update flash timer
            if enemy["flash_timer"] > 0:
                enemy["flash_timer"] -= 1
        return reward

    def _update_projectiles(self):
        reward = 0
        projectiles_to_keep = []
        enemies_to_keep = []

        for proj in self.projectiles:
            proj["pos"] += proj["vel"]
            
            hit_enemy = False
            for enemy in self.enemies:
                if proj["pos"].distance_to(enemy["pos"]) < self.ENEMY_SIZE:
                    # sfx: enemy_hit.wav
                    hit_enemy = True
                    is_headshot = enemy["health"] == 1
                    enemy["health"] -= 1
                    enemy["flash_timer"] = 3 # frames to flash
                    reward += 1.0 if is_headshot else 0.1
                    
                    if enemy["health"] <= 0:
                        # sfx: enemy_explode.wav
                        reward += 10.0
                        self.kills += 1
                        for _ in range(20):
                            self.particles.append(self._create_particle(enemy["pos"], random.choice(self.PARTICLE_COLORS)))
                    break 
            
            # Keep projectile if it's within bounds and hasn't hit anything
            if not hit_enemy and 0 < proj["pos"].x < self.WIDTH and 0 < proj["pos"].y < self.HEIGHT:
                projectiles_to_keep.append(proj)
        
        self.projectiles = projectiles_to_keep
        self.enemies = [e for e in self.enemies if e["health"] > 0]
        return reward

    def _update_particles(self):
        self.particles = [p for p in self.particles if p["life"] > 0]
        for p in self.particles:
            p["pos"] += p["vel"]
            p["life"] -= 1

    def _check_termination(self):
        if self.player_health <= 0:
            self.game_over = True
            return True
        if len(self.enemies) == 0:
            self.game_over = True
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        return False

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
            "health": self.player_health,
            "kills": self.kills,
            "enemies_remaining": len(self.enemies)
        }

    def _create_particle(self, pos, color, size_mult=1, life_mult=1, angle=None, spread=math.pi*2):
        if angle is None:
            angle = random.uniform(0, 2 * math.pi)
        else:
            angle += random.uniform(-spread/2, spread/2)
            
        speed = random.uniform(1, 4)
        vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
        return {
            "pos": pos.copy(),
            "vel": vel,
            "life": random.randint(10, 20) * life_mult,
            "color": color,
            "size": random.randint(2, 4) * size_mult
        }

    def _render_game(self):
        # Arena
        pygame.draw.rect(self.screen, self.COLOR_ARENA, (10, 10, self.WIDTH - 20, self.HEIGHT - 20), 2)
        
        # Particles
        for p in self.particles:
            alpha = int(255 * (p["life"] / 20))
            if alpha > 0:
                s = pygame.Surface((p["size"]*2, p["size"]*2), pygame.SRCALPHA)
                pygame.draw.circle(s, (*p["color"], min(alpha, 255)), (p["size"], p["size"]), p["size"])
                self.screen.blit(s, (int(p["pos"].x - p["size"]), int(p["pos"].y - p["size"])), special_flags=pygame.BLEND_RGBA_ADD)

        # Enemies
        for enemy in self.enemies:
            color = self.COLOR_ENEMY_FLASH if enemy["flash_timer"] > 0 else self.COLOR_ENEMY
            direction = self.player_pos - enemy["pos"]
            angle = direction.angle_to(pygame.Vector2(1, 0))
            
            points = []
            for i in range(3):
                a = math.radians(angle) + (i * 2 * math.pi / 3)
                p = enemy["pos"] + pygame.Vector2(math.cos(a), math.sin(a)) * self.ENEMY_SIZE
                points.append((int(p.x), int(p.y)))
            pygame.gfxdraw.aapolygon(self.screen, points, color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)

        # Projectiles
        for proj in self.projectiles:
            start = proj["pos"]
            end = proj["pos"] - proj["vel"].normalize() * 10
            pygame.draw.line(self.screen, self.COLOR_PROJECTILE, (int(start.x), int(start.y)), (int(end.x), int(end.y)), 3)

        # Player
        px, py = int(self.player_pos.x), int(self.player_pos.y)
        # Glow
        glow_surf = pygame.Surface((self.PLAYER_SIZE * 4, self.PLAYER_SIZE * 4), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, self.COLOR_PLAYER_GLOW, (self.PLAYER_SIZE * 2, self.PLAYER_SIZE * 2), self.PLAYER_SIZE * 2)
        self.screen.blit(glow_surf, (px - self.PLAYER_SIZE * 2, py - self.PLAYER_SIZE * 2), special_flags=pygame.BLEND_RGBA_ADD)
        # Body
        player_rect = pygame.Rect(px - self.PLAYER_SIZE/2, py - self.PLAYER_SIZE/2, self.PLAYER_SIZE, self.PLAYER_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect)
        # Turret/Aim indicator
        turret_end_x = px + math.cos(self.player_aim_angle) * (self.PLAYER_SIZE)
        turret_end_y = py + math.sin(self.player_aim_angle) * (self.PLAYER_SIZE)
        pygame.draw.line(self.screen, self.COLOR_TEXT, (px, py), (int(turret_end_x), int(turret_end_y)), 3)

    def _render_ui(self):
        # Health Bar
        health_ratio = max(0, self.player_health / self.PLAYER_MAX_HEALTH)
        bar_width = 200
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BG, (20, 20, bar_width, 20))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH, (20, 20, bar_width * health_ratio, 20))
        
        # Score
        score_text = self.font_small.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 20, 20))

        # Enemy Count
        enemy_text = self.font_small.render(f"ENEMIES: {len(self.enemies)}/{self.TOTAL_ENEMIES}", True, self.COLOR_TEXT)
        self.screen.blit(enemy_text, (self.WIDTH/2 - enemy_text.get_width()/2, 20))

        if self.game_over:
            if self.player_health <= 0:
                end_text = self.font_large.render("GAME OVER", True, self.COLOR_ENEMY)
            else:
                end_text = self.font_large.render("VICTORY", True, self.COLOR_HEALTH)
            self.screen.blit(end_text, (self.WIDTH/2 - end_text.get_width()/2, self.HEIGHT/2 - end_text.get_height()/2))

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
        
        # Test state guarantees
        assert self.player_health <= self.PLAYER_MAX_HEALTH
        assert len(self.enemies) <= self.TOTAL_ENEMIES
        assert self.score >= 0
        assert len(self.projectiles) <= 50 # Reasonable limit

        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    # Set this to False to run in headless mode for testing
    human_playing = True
    
    if human_playing:
        # Un-set the dummy video driver to allow display
        if "SDL_VIDEODRIVER" in os.environ:
            del os.environ["SDL_VIDEODRIVER"]
        env = GameEnv(render_mode="human")
        screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        pygame.display.set_caption("Robot Arena")
    else:
        env = GameEnv()

    obs, info = env.reset()
    done = False
    
    # --- Main Game Loop for Human Player ---
    if human_playing:
        while not done:
            # Action defaults
            movement, space, shift = 0, 0, 0

            # Pygame event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
            
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            if keys[pygame.K_SPACE]: space = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

            action = [movement, space, shift]
            
            # Step the environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Render the observation to the display
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()

            # Cap the frame rate
            env.clock.tick(30)
        
        print(f"Game Over! Final Score: {info['score']}")
        pygame.time.wait(2000)
        env.close()
    else:
        # --- Example of running the environment headlessly ---
        print("Running headless test...")
        total_reward = 0
        for _ in range(1000):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                print(f"Episode finished. Total reward: {total_reward}, Info: {info}")
                obs, info = env.reset()
                total_reward = 0
        env.close()