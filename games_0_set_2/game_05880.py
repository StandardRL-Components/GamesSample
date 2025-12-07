
# Generated: 2025-08-28T06:21:58.166695
# Source Brief: brief_05880.md
# Brief Index: 5880

        
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move. Hold space to fire your weapon. Press shift to reload."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Survive hordes of zombies in an isometric arena for five minutes. Manage your ammo and keep moving!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 3000
        
        # Colors
        self.COLOR_BG = (34, 32, 52)
        self.COLOR_GRID = (44, 42, 62)
        self.COLOR_WALL = (80, 80, 100)
        self.COLOR_PLAYER = (50, 255, 50)
        self.COLOR_ZOMBIE = (255, 50, 50)
        self.COLOR_PROJECTILE = (255, 255, 0)
        self.COLOR_SHADOW = (20, 18, 32, 128)
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.COLOR_UI_BG = (50, 50, 70)
        self.COLOR_HEALTH = (220, 30, 30)
        self.COLOR_AMMO = (220, 180, 30)

        # Player settings
        self.PLAYER_SPEED = 4
        self.PLAYER_MAX_HEALTH = 100
        self.PLAYER_MAX_AMMO = 20
        self.SHOOT_COOLDOWN = 3  # steps
        self.RELOAD_TIME = 30  # steps

        # Zombie settings
        self.ZOMBIE_SPEED = 1.5
        self.ZOMBIE_DAMAGE = 20
        self.ZOMBIE_SPAWN_INTERVAL = 150 # steps
        
        # Entity sizes (using pseudo-isometric diamond shapes)
        self.PLAYER_W, self.PLAYER_H = 16, 16
        self.ZOMBIE_W, self.ZOMBIE_H = 16, 16

        # World boundaries
        self.ARENA_MARGIN = 30
        self.ARENA_BOUNDS = pygame.Rect(self.ARENA_MARGIN, self.ARENA_MARGIN, 
                                        self.WIDTH - 2 * self.ARENA_MARGIN, 
                                        self.HEIGHT - 2 * self.ARENA_MARGIN)

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_s = pygame.font.SysFont("monospace", 16)
        self.font_m = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_l = pygame.font.SysFont("monospace", 48, bold=True)
        
        # Initialize state variables
        self.reset()

        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        
        self.player_pos = [self.WIDTH / 2, self.HEIGHT / 2]
        self.player_health = self.PLAYER_MAX_HEALTH
        self.player_ammo = self.PLAYER_MAX_AMMO
        self.player_last_move_dir = np.array([0, -1]) # Default aim up
        self.player_reloading_timer = 0
        self.player_shoot_cooldown = 0
        
        self.zombies = []
        self.projectiles = []
        self.particles = []
        
        self.zombie_spawn_timer = 0
        self.screen_shake = 0
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0.01  # Small reward for surviving a step

        self._handle_input(action)
        self._update_player()
        self._update_zombies()
        self._update_projectiles()
        self._update_particles()
        
        kill_reward = self._handle_collisions()
        reward += kill_reward
        
        self._spawn_zombies()
        
        self.steps += 1
        
        terminated = False
        if self.player_health <= 0:
            terminated = True
            self.game_over = True
            reward = -100.0
            # SFX: game_over.wav
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
            self.win = True
            reward += 100.0
            # SFX: victory.wav

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        is_reloading = self.player_reloading_timer > 0

        # Movement
        move_vec = np.array([0.0, 0.0])
        if not is_reloading:
            if movement == 1: move_vec[1] = -1 # Up
            elif movement == 2: move_vec[1] = 1  # Down
            elif movement == 3: move_vec[0] = -1 # Left
            elif movement == 4: move_vec[0] = 1  # Right
        
        if np.linalg.norm(move_vec) > 0:
            self.player_pos[0] += move_vec[0] * self.PLAYER_SPEED
            self.player_pos[1] += move_vec[1] * self.PLAYER_SPEED
            self.player_last_move_dir = move_vec
        
        # Clamp player position to arena
        self.player_pos[0] = np.clip(self.player_pos[0], self.ARENA_BOUNDS.left, self.ARENA_BOUNDS.right)
        self.player_pos[1] = np.clip(self.player_pos[1], self.ARENA_BOUNDS.top, self.ARENA_BOUNDS.bottom)
        
        # Shooting
        if space_held and self.player_ammo > 0 and self.player_shoot_cooldown == 0 and not is_reloading:
            self.player_ammo -= 1
            self.player_shoot_cooldown = self.SHOOT_COOLDOWN
            
            proj_pos = list(self.player_pos)
            proj_dir = self.player_last_move_dir / np.linalg.norm(self.player_last_move_dir)
            self.projectiles.append({"pos": proj_pos, "dir": proj_dir})
            
            self._create_muzzle_flash(proj_pos, proj_dir)
            # SFX: gunshot.wav

        # Reloading
        if shift_held and self.player_ammo < self.PLAYER_MAX_AMMO and not is_reloading:
            self.player_reloading_timer = self.RELOAD_TIME
            # SFX: reload.wav
            
    def _update_player(self):
        if self.player_shoot_cooldown > 0:
            self.player_shoot_cooldown -= 1
        if self.player_reloading_timer > 0:
            self.player_reloading_timer -= 1
            if self.player_reloading_timer == 0:
                self.player_ammo = self.PLAYER_MAX_AMMO
                # SFX: reload_complete.wav
                
    def _update_zombies(self):
        for z in self.zombies:
            direction = np.array(self.player_pos) - np.array(z["pos"])
            dist = np.linalg.norm(direction)
            if dist > 1:
                direction /= dist
            z["pos"] += direction * self.ZOMBIE_SPEED
            
    def _update_projectiles(self):
        self.projectiles = [p for p in self.projectiles if self.ARENA_BOUNDS.collidepoint(p["pos"])]
        for p in self.projectiles:
            p["pos"] += p["dir"] * 10 # Projectile speed

    def _update_particles(self):
        self.particles = [p for p in self.particles if p["life"] > 0]
        for p in self.particles:
            p["pos"] += p["vel"]
            p["life"] -= 1
            p["size"] = max(0, p["size"] - p["decay"])

    def _handle_collisions(self):
        kill_reward = 0
        
        # Projectiles vs Zombies
        dead_zombies = []
        used_projectiles = []
        for i, p in enumerate(self.projectiles):
            for j, z in enumerate(self.zombies):
                if j in dead_zombies: continue
                dist = np.linalg.norm(np.array(p["pos"]) - np.array(z["pos"]))
                if dist < (self.ZOMBIE_W / 2 + 4): # 4 is projectile radius
                    dead_zombies.append(j)
                    used_projectiles.append(i)
                    kill_reward += 1.0
                    self.score += 10
                    self._create_blood_splatter(z["pos"])
                    # SFX: zombie_death.wav
                    break # Projectile can only hit one zombie

        self.projectiles = [p for i, p in enumerate(self.projectiles) if i not in used_projectiles]
        self.zombies = [z for i, z in enumerate(self.zombies) if i not in dead_zombies]
        
        # Zombies vs Player
        zombies_to_remove = []
        for i, z in enumerate(self.zombies):
            dist = np.linalg.norm(np.array(self.player_pos) - np.array(z["pos"]))
            if dist < (self.PLAYER_W / 2 + self.ZOMBIE_W / 2):
                self.player_health = max(0, self.player_health - self.ZOMBIE_DAMAGE)
                zombies_to_remove.append(i)
                self.screen_shake = 5
                self._create_blood_splatter(self.player_pos, count=15, color=(0, 200, 0))
                # SFX: player_hit.wav

        if zombies_to_remove:
            self.zombies = [z for i, z in enumerate(self.zombies) if i not in zombies_to_remove]
            
        return kill_reward

    def _spawn_zombies(self):
        self.zombie_spawn_timer += 1
        if self.zombie_spawn_timer >= self.ZOMBIE_SPAWN_INTERVAL:
            self.zombie_spawn_timer = 0
            num_to_spawn = 1 + self.steps // 600
            
            for _ in range(num_to_spawn):
                edge = self.np_random.integers(4)
                if edge == 0: # Top
                    x = self.np_random.uniform(0, self.WIDTH)
                    y = -self.ZOMBIE_H
                elif edge == 1: # Bottom
                    x = self.np_random.uniform(0, self.WIDTH)
                    y = self.HEIGHT + self.ZOMBIE_H
                elif edge == 2: # Left
                    x = -self.ZOMBIE_W
                    y = self.np_random.uniform(0, self.HEIGHT)
                else: # Right
                    x = self.WIDTH + self.ZOMBIE_W
                    y = self.np_random.uniform(0, self.HEIGHT)
                
                self.zombies.append({"pos": np.array([x, y])})

    def _get_observation(self):
        # Apply screen shake
        shake_offset = (0, 0)
        if self.screen_shake > 0:
            self.screen_shake -= 1
            shake_offset = (self.np_random.integers(-self.screen_shake, self.screen_shake + 1),
                            self.np_random.integers(-self.screen_shake, self.screen_shake + 1))

        # Create a temporary surface to draw on, allowing for the shake offset
        temp_surface = pygame.Surface((self.WIDTH, self.HEIGHT))
        
        # Clear screen with background
        temp_surface.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game(temp_surface)
        
        # Blit the shaken game scene to the main screen
        self.screen.blit(temp_surface, shake_offset)
        
        # Render UI overlay (not affected by shake)
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self, surface):
        self._draw_grid(surface)
        pygame.draw.rect(surface, self.COLOR_WALL, self.ARENA_BOUNDS, 2)
        
        render_list = []
        render_list.append({"y": self.player_pos[1], "type": "player"})
        for z in self.zombies:
            render_list.append({"y": z["pos"][1], "type": "zombie", "data": z})
            
        render_list.sort(key=lambda item: item["y"])
        
        # Draw shadows first
        for item in render_list:
            if item["type"] == "player":
                self._draw_shadow(surface, self.player_pos, self.PLAYER_W)
            elif item["type"] == "zombie":
                self._draw_shadow(surface, item["data"]["pos"], self.ZOMBIE_W)

        # Draw entities
        for item in render_list:
            if item["type"] == "player":
                self._draw_diamond(surface, self.player_pos, self.PLAYER_W, self.PLAYER_H, self.COLOR_PLAYER)
            elif item["type"] == "zombie":
                self._draw_diamond(surface, item["data"]["pos"], self.ZOMBIE_W, self.ZOMBIE_H, self.COLOR_ZOMBIE)

        for p in self.projectiles:
            pygame.gfxdraw.filled_circle(surface, int(p["pos"][0]), int(p["pos"][1]), 3, self.COLOR_PROJECTILE)
            pygame.gfxdraw.aacircle(surface, int(p["pos"][0]), int(p["pos"][1]), 3, self.COLOR_PROJECTILE)

        for p in self.particles:
            pygame.draw.circle(surface, p["color"], (int(p["pos"][0]), int(p["pos"][1])), int(p["size"]))

    def _render_ui(self):
        # Health bar
        health_pct = self.player_health / self.PLAYER_MAX_HEALTH
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, (10, 10, 150, 20))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH, (10, 10, 150 * health_pct, 20))
        
        # Ammo bar
        ammo_pct = self.player_ammo / self.PLAYER_MAX_AMMO
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, (10, 35, 150, 20))
        pygame.draw.rect(self.screen, self.COLOR_AMMO, (10, 35, 150 * ammo_pct, 20))
        
        # Timer
        time_left = max(0, self.MAX_STEPS - self.steps)
        time_text = self.font_m.render(f"TIME: {time_left}", True, self.COLOR_UI_TEXT)
        self.screen.blit(time_text, (self.WIDTH - time_text.get_width() - 10, 10))

        # Score
        score_text = self.font_m.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.WIDTH // 2 - score_text.get_width() // 2, 10))

        # Reloading indicator
        if self.player_reloading_timer > 0:
            reload_text = self.font_m.render("RELOADING...", True, self.COLOR_AMMO)
            self.screen.blit(reload_text, (self.WIDTH // 2 - reload_text.get_width() // 2, self.HEIGHT - 40))

        # Game Over / Win message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            msg = "YOU SURVIVED!" if self.win else "GAME OVER"
            msg_text = self.font_l.render(msg, True, self.COLOR_UI_TEXT)
            self.screen.blit(msg_text, (self.WIDTH // 2 - msg_text.get_width() // 2, self.HEIGHT // 2 - msg_text.get_height() // 2))

    def _draw_grid(self, surface):
        for x in range(0, self.WIDTH, 40):
            pygame.draw.line(surface, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, 40):
            pygame.draw.line(surface, self.COLOR_GRID, (0, y), (self.WIDTH, y))

    def _draw_diamond(self, surface, pos, w, h, color):
        x, y = int(pos[0]), int(pos[1])
        points = [(x, y - h // 2), (x + w // 2, y), (x, y + h // 2), (x - w // 2, y)]
        pygame.gfxdraw.filled_polygon(surface, points, color)
        pygame.gfxdraw.aapolygon(surface, points, color)

    def _draw_shadow(self, surface, pos, w):
        shadow_surface = pygame.Surface((w, w // 2), pygame.SRCALPHA)
        pygame.draw.ellipse(shadow_surface, self.COLOR_SHADOW, (0, 0, w, w // 2))
        surface.blit(shadow_surface, (int(pos[0] - w / 2), int(pos[1] + 4)))
        
    def _create_muzzle_flash(self, pos, direction):
        for _ in range(10):
            angle = math.atan2(direction[1], direction[0]) + self.np_random.uniform(-0.5, 0.5)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                "pos": list(pos), "vel": vel, "life": 5, "size": self.np_random.uniform(1, 4),
                "color": (255, self.np_random.integers(150, 255), 0), "decay": 0.8
            })

    def _create_blood_splatter(self, pos, count=20, color=(200, 0, 0)):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(0.5, 3)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                "pos": list(pos), "vel": vel, "life": self.np_random.integers(10, 25),
                "size": self.np_random.uniform(2, 6), "color": color, "decay": 0.2
            })

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.player_health,
            "ammo": self.player_ammo
        }

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
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.init()
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Zombie Survival")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    print("\n" + "="*30)
    print(env.game_description)
    print(env.user_guide)
    print("="*30 + "\n")
    
    while running:
        # Get human input
        movement = 0 # none
        space = 0 # released
        shift = 0 # released
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
            
        action = [movement, space, shift]
        
        # Environment step
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                total_reward = 0

        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Score: {info['score']}, Steps: {info['steps']}")
            print("Press 'R' to play again or close the window to quit.")
            
            # Wait for reset command
            wait_for_reset = True
            while wait_for_reset:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        wait_for_reset = False
                        running = False
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                        print("Resetting environment.")
                        obs, info = env.reset()
                        total_reward = 0
                        wait_for_reset = False
                clock.tick(30)

        clock.tick(30) # Limit frame rate for human play
        
    env.close()