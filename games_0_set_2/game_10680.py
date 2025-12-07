import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:45:56.360742
# Source Brief: brief_00680.md
# Brief Index: 680
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
from collections import deque

class GameEnv(gym.Env):
    """
    A Gymnasium environment for a cyberpunk-themed top-down shooter.
    The player pilots a vehicle through a fractal city, destroying enemy
    siege weapons. Aiming is based on the player's momentum, and destroying
    enemies can trigger satisfying chain-reaction explosions.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Pilot a cyber-vehicle through a fractal city in this top-down shooter. "
        "Use your momentum to aim and destroy enemy siege weapons in chain-reaction explosions."
    )
    user_guide = (
        "Use the arrow keys (↑↓←→) to move your vehicle. Press space to fire. "
        "Your shots are aimed based on your vehicle's momentum."
    )
    auto_advance = True

    # --- CONSTANTS ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 5000

    # Colors
    COLOR_BG = (15, 10, 25)
    COLOR_GRID = (30, 20, 50)
    COLOR_PLAYER = (0, 191, 255)
    COLOR_PLAYER_GLOW = (0, 191, 255, 50)
    COLOR_ENEMY = (255, 50, 50)
    COLOR_ENEMY_GLOW = (255, 50, 50, 50)
    COLOR_PLAYER_PROJ = (173, 216, 230)
    COLOR_ENEMY_PROJ = (255, 180, 180)
    COLOR_TEXT = (220, 220, 220)
    COLOR_HEALTH_BAR = (0, 255, 127)
    COLOR_HEALTH_BAR_BG = (139, 0, 0)
    
    # Player
    PLAYER_SIZE = 12
    PLAYER_ACCEL = 0.6
    PLAYER_FRICTION = 0.94
    PLAYER_MAX_HEALTH = 100
    PLAYER_FIRE_COOLDOWN = 8  # steps

    # Projectiles
    PROJ_SPEED = 8.0
    PROJ_SIZE = 3
    PROJ_LIFETIME = 80 # steps

    # Enemy
    ENEMY_SIZE = 15
    ENEMY_BASE_HEALTH = 30
    ENEMY_BASE_FIRE_RATE = 100 # steps

    # Effects
    EXPLOSION_RADIUS = 60
    EXPLOSION_DAMAGE = 50
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.render_mode = render_mode
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 16)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        
        # State variables are initialized in reset()
        self.player_pos = None
        self.player_vel = None
        self.player_health = None
        self.player_fire_cooldown = None
        self.last_move_vector = None
        self.enemies = None
        self.player_projectiles = None
        self.enemy_projectiles = None
        self.particles = None
        self.steps = None
        self.score = None
        self.wave = None
        self.game_over = None
        self.fractal_bg = None
        self.last_space_state = None

        # The validate_implementation method requires a reset first
        # self.reset()
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.wave = 1
        self.game_over = False
        
        self.player_pos = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2], dtype=np.float32)
        self.player_vel = np.array([0.0, 0.0], dtype=np.float32)
        self.player_health = self.PLAYER_MAX_HEALTH
        self.player_fire_cooldown = 0
        self.last_move_vector = np.array([0.0, -1.0], dtype=np.float32) # Default to up
        self.last_space_state = 0

        self.enemies = []
        self.player_projectiles = deque()
        self.enemy_projectiles = deque()
        self.particles = deque()
        
        self.fractal_bg = self._create_fractal_surface()
        self._spawn_wave()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0.0
        self.steps += 1
        
        # --- Action Handling ---
        movement, space_button, _ = action
        space_pressed = space_button == 1 and self.last_space_state == 0
        self.last_space_state = space_button

        # --- Player Movement ---
        self._handle_player_movement(movement)
        
        # --- Player Firing ---
        if space_pressed and self.player_fire_cooldown <= 0:
            self._fire_player_projectile()
            # sfx: player_shoot.wav
            
        # --- Update Game State ---
        self._update_cooldowns()
        self._update_projectiles()
        self._update_enemies()
        self._update_particles()
        
        # --- Collision Detection & Resolution ---
        step_rewards = self._handle_collisions()
        reward += step_rewards
        
        # --- Wave Management ---
        if not self.enemies:
            reward += 100
            self.wave += 1
            self._spawn_wave()
            # sfx: wave_cleared.wav
            
        # --- Termination Check ---
        terminated = False
        if self.player_health <= 0:
            reward -= 100
            terminated = True
            # sfx: player_death.wav
        if self.steps >= self.MAX_STEPS:
            reward -= 100 # Penalize for running out of time
            terminated = True
        
        self.game_over = terminated
        self.score += reward
        
        # Add small survival reward
        reward += 0.01

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_player_movement(self, movement_action):
        move_vec = np.array([0, 0], dtype=np.float32)
        if movement_action == 1: move_vec[1] = -1 # Up
        elif movement_action == 2: move_vec[1] = 1  # Down
        elif movement_action == 3: move_vec[0] = -1 # Left
        elif movement_action == 4: move_vec[0] = 1  # Right
        
        if np.linalg.norm(move_vec) > 0:
            self.last_move_vector = move_vec
            self.player_vel += move_vec * self.PLAYER_ACCEL
        
        self.player_vel *= self.PLAYER_FRICTION
        self.player_pos += self.player_vel
        
        # World wrapping
        self.player_pos[0] %= self.SCREEN_WIDTH
        self.player_pos[1] %= self.SCREEN_HEIGHT

    def _fire_player_projectile(self):
        self.player_fire_cooldown = self.PLAYER_FIRE_COOLDOWN
        
        # Aim with momentum, or last direction if still
        fire_dir = self.player_vel.copy()
        if np.linalg.norm(fire_dir) < 0.5:
            fire_dir = self.last_move_vector
        
        if np.linalg.norm(fire_dir) > 0:
            norm_dir = fire_dir / np.linalg.norm(fire_dir)
            proj_vel = norm_dir * self.PROJ_SPEED
            self.player_projectiles.append({
                "pos": self.player_pos.copy(),
                "vel": proj_vel,
                "life": self.PROJ_LIFETIME
            })

    def _update_cooldowns(self):
        if self.player_fire_cooldown > 0:
            self.player_fire_cooldown -= 1
        for enemy in self.enemies:
            if enemy["cooldown"] > 0:
                enemy["cooldown"] -= 1

    def _update_projectiles(self):
        for proj_list in [self.player_projectiles, self.enemy_projectiles]:
            i = len(proj_list) - 1
            while i >= 0:
                proj = proj_list[i]
                proj["pos"] += proj["vel"]
                proj["life"] -= 1
                
                # World wrapping
                proj["pos"][0] %= self.SCREEN_WIDTH
                proj["pos"][1] %= self.SCREEN_HEIGHT
                
                if proj["life"] <= 0:
                    proj_list.remove(proj)
                i -= 1

    def _update_enemies(self):
        for enemy in self.enemies:
            if enemy["cooldown"] <= 0:
                # Aim at player
                direction = self.player_pos - enemy["pos"]
                dist = np.linalg.norm(direction)
                if dist > 0:
                    direction /= dist
                    # Add some inaccuracy
                    angle = math.atan2(direction[1], direction[0]) + self.np_random.uniform(-0.1, 0.1)
                    inaccurate_dir = np.array([math.cos(angle), math.sin(angle)])
                    
                    proj_speed = self.PROJ_SPEED * 0.7 + (self.wave // 5) * 0.05
                    self.enemy_projectiles.append({
                        "pos": enemy["pos"].copy(),
                        "vel": inaccurate_dir * proj_speed,
                        "life": self.PROJ_LIFETIME
                    })
                    enemy["cooldown"] = enemy["fire_rate"]
                    # sfx: enemy_shoot.wav

    def _update_particles(self):
        i = len(self.particles) - 1
        while i >= 0:
            p = self.particles[i]
            p["pos"] += p["vel"]
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)
            i -= 1

    def _handle_collisions(self):
        reward = 0
        explosions = []
        
        # Player projectiles vs Enemies
        destroyed_this_step = 0
        for proj in list(self.player_projectiles):
            for enemy in self.enemies:
                if np.linalg.norm(proj["pos"] - enemy["pos"]) < self.ENEMY_SIZE:
                    enemy["health"] -= 20
                    reward += 1.0 # Hit reward
                    self._create_impact_particles(proj["pos"])
                    if proj in self.player_projectiles: self.player_projectiles.remove(proj)
                    # sfx: enemy_hit.wav
                    
                    if enemy["health"] <= 0:
                        reward += 5.0 # Destroy reward
                        explosions.append({"pos": enemy["pos"].copy(), "radius": self.EXPLOSION_RADIUS, "damage": self.EXPLOSION_DAMAGE})
                        self.enemies.remove(enemy)
                        destroyed_this_step += 1
                        # sfx: explosion.wav
                    break
        
        if destroyed_this_step > 1:
            reward += 2.0 * (destroyed_this_step - 1) # Chain reaction bonus

        # Explosions vs Enemies
        for explosion in explosions:
            self._create_explosion_particles(explosion["pos"])
            for enemy in list(self.enemies):
                if np.linalg.norm(explosion["pos"] - enemy["pos"]) < explosion["radius"]:
                    enemy["health"] -= explosion["damage"]
                    if enemy["health"] <= 0:
                        reward += 5.0 # Bonus for chain reaction kills
                        explosions.append({"pos": enemy["pos"].copy(), "radius": self.EXPLOSION_RADIUS, "damage": self.EXPLOSION_DAMAGE})
                        self.enemies.remove(enemy)
                        # sfx: explosion.wav

        # Enemy projectiles vs Player
        for proj in list(self.enemy_projectiles):
            if np.linalg.norm(proj["pos"] - self.player_pos) < self.PLAYER_SIZE:
                self.player_health -= 10
                self._create_impact_particles(proj["pos"], is_player_impact=True)
                if proj in self.enemy_projectiles: self.enemy_projectiles.remove(proj)
                # sfx: player_hit.wav
        
        return reward

    def _spawn_wave(self):
        num_enemies = 2 + self.wave
        for _ in range(num_enemies):
            while True:
                pos = np.array([self.np_random.uniform(0, self.SCREEN_WIDTH), self.np_random.uniform(0, self.SCREEN_HEIGHT)], dtype=np.float32)
                if np.linalg.norm(pos - self.player_pos) > 150: # Don't spawn on player
                    break
            self.enemies.append({
                "pos": pos,
                "health": self.ENEMY_BASE_HEALTH + self.wave * 5,
                "fire_rate": max(30, self.ENEMY_BASE_FIRE_RATE - self.wave * 4),
                "cooldown": self.np_random.integers(0, self.ENEMY_BASE_FIRE_RATE + 1)
            })

    def _create_fractal_surface(self):
        surf = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        surf.fill(self.COLOR_BG)
        
        # Draw grid
        for i in range(0, self.SCREEN_WIDTH, 40):
            pygame.draw.line(surf, self.COLOR_GRID, (i, 0), (i, self.SCREEN_HEIGHT))
        for i in range(0, self.SCREEN_HEIGHT, 40):
            pygame.draw.line(surf, self.COLOR_GRID, (0, i), (self.SCREEN_WIDTH, i))

        # Draw recursive fractal "buildings"
        def draw_rects(x, y, w, h, depth):
            if depth <= 0 or w < 20 or h < 20:
                return
            
            color = (
                self.COLOR_BG[0] + depth * 5,
                self.COLOR_BG[1] + depth * 3,
                self.COLOR_BG[2] + depth * 8
            )
            pygame.draw.rect(surf, color, (x, y, w, h), 1)
            
            if self.np_random.random() > 0.5: # Split vertically
                split = self.np_random.uniform(0.3, 0.7)
                draw_rects(x, y, w * split, h, depth - 1)
                draw_rects(x + w * split, y, w * (1-split), h, depth - 1)
            else: # Split horizontally
                split = self.np_random.uniform(0.3, 0.7)
                draw_rects(x, y, w, h * split, depth - 1)
                draw_rects(x, y + h * split, w, h * (1-split), depth - 1)

        for _ in range(5):
             draw_rects(
                 self.np_random.uniform(-100, self.SCREEN_WIDTH),
                 self.np_random.uniform(-100, self.SCREEN_HEIGHT),
                 self.np_random.uniform(100, 400),
                 self.np_random.uniform(100, 400),
                 4
            )
        return surf

    def _create_explosion_particles(self, pos):
        colors = [(255, 165, 0), (255, 255, 0), (255, 69, 0)]
        for _ in range(50):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            self.particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "life": self.np_random.integers(20, 41),
                "color": colors[self.np_random.integers(len(colors))],
                "radius": self.np_random.uniform(1, 4)
            })

    def _create_impact_particles(self, pos, is_player_impact=False):
        num_particles = 5 if is_player_impact else 15
        color = self.COLOR_PLAYER_PROJ if is_player_impact else self.COLOR_ENEMY_PROJ
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(0.5, 2)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            self.particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "life": self.np_random.integers(10, 21),
                "color": color,
                "radius": self.np_random.uniform(1, 2)
            })

    def _get_observation(self):
        self.screen.blit(self.fractal_bg, (0, 0))
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Particles
        for p in self.particles:
            pygame.gfxdraw.filled_circle(
                self.screen, int(p["pos"][0]), int(p["pos"][1]),
                int(p["radius"] * (p["life"] / 20.0)), p["color"]
            )
        
        # Projectiles
        for proj in self.enemy_projectiles:
            pygame.gfxdraw.filled_circle(self.screen, int(proj["pos"][0]), int(proj["pos"][1]), self.PROJ_SIZE, self.COLOR_ENEMY_PROJ)
        for proj in self.player_projectiles:
            pygame.gfxdraw.filled_circle(self.screen, int(proj["pos"][0]), int(proj["pos"][1]), self.PROJ_SIZE, self.COLOR_PLAYER_PROJ)
        
        # Enemies
        for enemy in self.enemies:
            x, y = int(enemy["pos"][0]), int(enemy["pos"][1])
            pygame.gfxdraw.filled_circle(self.screen, x, y, self.ENEMY_SIZE + 5, self.COLOR_ENEMY_GLOW)
            pygame.gfxdraw.filled_circle(self.screen, x, y, self.ENEMY_SIZE, self.COLOR_ENEMY)
            pygame.gfxdraw.aacircle(self.screen, x, y, self.ENEMY_SIZE, self.COLOR_ENEMY)

        # Player
        px, py = int(self.player_pos[0]), int(self.player_pos[1])
        pygame.gfxdraw.filled_circle(self.screen, px, py, self.PLAYER_SIZE + 8, self.COLOR_PLAYER_GLOW)
        pygame.gfxdraw.filled_circle(self.screen, px, py, self.PLAYER_SIZE, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, px, py, self.PLAYER_SIZE, self.COLOR_PLAYER)

        # Player momentum indicator
        if np.linalg.norm(self.player_vel) > 0.1:
            end_pos = self.player_pos + self.player_vel * 5
            pygame.draw.line(self.screen, self.COLOR_PLAYER, (px, py), (int(end_pos[0]), int(end_pos[1])), 2)

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Wave
        wave_text = self.font_large.render(f"WAVE: {self.wave}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (self.SCREEN_WIDTH - wave_text.get_width() - 10, 10))

        # Timer
        time_left = self.MAX_STEPS - self.steps
        time_text = self.font_small.render(f"TIME: {time_left}", True, self.COLOR_TEXT)
        self.screen.blit(time_text, (self.SCREEN_WIDTH // 2 - time_text.get_width() // 2, 30))

        # Health Bar
        health_ratio = max(0, self.player_health / self.PLAYER_MAX_HEALTH)
        bar_width = 200
        bar_height = 15
        bar_x = self.SCREEN_WIDTH // 2 - bar_width // 2
        bar_y = 10
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (bar_x, bar_y, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (bar_x, bar_y, int(bar_width * health_ratio), bar_height))
        
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave,
            "player_health": self.player_health,
            "enemies_left": len(self.enemies)
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # For headed mode, comment out the os.environ line at the top
    # os.environ["SDL_VIDEODRIVER"] = "x11" 
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Game loop for human player
    running = True
    total_reward = 0
    
    # Map keyboard keys to actions
    key_map = {
        pygame.K_w: 1, pygame.K_UP: 1,
        pygame.K_s: 2, pygame.K_DOWN: 2,
        pygame.K_a: 3, pygame.K_LEFT: 3,
        pygame.K_d: 4, pygame.K_RIGHT: 4
    }

    # Pygame setup for rendering
    pygame.display.set_caption("Fractal Siege")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    clock = pygame.time.Clock()

    while running:
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
                done = False

        if done:
            # Display game over message
            font = pygame.font.SysFont("Consolas", 40, bold=True)
            text = font.render("GAME OVER", True, (255, 0, 0))
            text_rect = text.get_rect(center=(env.SCREEN_WIDTH/2, env.SCREEN_HEIGHT/2 - 20))
            screen.blit(text, text_rect)
            
            font_small = pygame.font.SysFont("Consolas", 20)
            text_small = font_small.render("Press 'R' to Restart", True, (200, 200, 200))
            text_small_rect = text_small.get_rect(center=(env.SCREEN_WIDTH/2, env.SCREEN_HEIGHT/2 + 20))
            screen.blit(text_small, text_small_rect)
            
            pygame.display.flip()
            continue

        # --- Action Selection ---
        keys = pygame.key.get_pressed()
        
        movement = 0 # No-op
        for key, move_action in key_map.items():
            if keys[key]:
                movement = move_action
                break # Prioritize first key in map
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        # --- Rendering ---
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Frame Rate ---
        clock.tick(30)

    env.close()