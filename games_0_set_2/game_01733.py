import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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
    """
    A top-down tank arena game. Control a tank to destroy the enemy before it destroys you.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Short, user-facing control string:
    user_guide = (
        "Controls: ↑↓←→ to move. Hold shift for no effect (reserved). Press space to fire."
    )

    # Short, user-facing description of the game:
    game_description = (
        "Control a tank in a top-down arena. Strategically maneuver and fire to destroy the enemy tank."
    )

    # Frames auto-advance at 30fps for real-time gameplay.
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 1500  # 50 seconds at 30fps

    # Colors
    COLOR_BG = (25, 35, 45)
    COLOR_WALL = (180, 190, 200)
    COLOR_PLAYER = (50, 205, 50)  # LimeGreen
    COLOR_ENEMY = (220, 20, 60)   # Crimson
    COLOR_PLAYER_PROJ = (173, 255, 47) # GreenYellow
    COLOR_ENEMY_PROJ = (255, 105, 180) # HotPink
    COLOR_EXPLOSION = (255, 165, 0) # Orange
    COLOR_TEXT = (240, 240, 240)
    COLOR_HEALTH_GREEN = (34, 139, 34)
    COLOR_HEALTH_RED = (139, 0, 0)
    
    # Game parameters
    TANK_SIZE = 24
    TANK_ACCEL = 0.6
    TANK_FRICTION = 0.92
    TANK_MAX_SPEED = 4.0
    TURRET_LENGTH = 18
    
    PROJ_SPEED = 8.0
    PROJ_RADIUS = 5
    
    PLAYER_FIRE_COOLDOWN = 8 # frames
    PLAYER_MAX_HEALTH = 3
    PLAYER_MAX_AMMO = 5

    ENEMY_FIRE_COOLDOWN = 45 # frames
    ENEMY_MAX_HEALTH = 3
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)
        
        # Internal state variables
        self.player = None
        self.enemy = None
        self.player_projectiles = []
        self.enemy_projectiles = []
        self.explosions = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.winner = None # 'player' or 'enemy'
        self.np_random = None

        # Call reset to initialize state
        # self.reset() # Deferring to allow seed setting before first reset

        # Validate implementation after initialization
        # self.validate_implementation() # Deferring to avoid premature rendering

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.winner = None

        self.player = self._Tank(
            pos=pygame.Vector2(self.WIDTH * 0.25, self.HEIGHT / 2),
            color=self.COLOR_PLAYER,
            max_health=self.PLAYER_MAX_HEALTH,
            max_ammo=self.PLAYER_MAX_AMMO
        )
        
        self.enemy = self._Tank(
            pos=pygame.Vector2(self.WIDTH * 0.75, self.HEIGHT / 2),
            color=self.COLOR_ENEMY,
            max_health=self.ENEMY_MAX_HEALTH
        )
        self.enemy.patrol_dir = 1

        self.player_projectiles.clear()
        self.enemy_projectiles.clear()
        self.explosions.clear()

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        reward = 0
        terminated = False
        
        if not self.game_over:
            # --- Update game logic ---
            self.steps += 1
            
            # Distance before movement for reward calculation
            dist_before = self.player.pos.distance_to(self.enemy.pos)

            self._handle_player_input(movement, space_held)
            self._update_enemy_ai()
            self._update_tanks()
            
            hit_enemy, was_hit = self._update_projectiles()
            if hit_enemy:
                reward += 25.0
            if was_hit:
                reward -= 25.0

            self._update_explosions()

            # Distance-based reward
            dist_after = self.player.pos.distance_to(self.enemy.pos)
            if dist_after < dist_before:
                reward += 0.1
            else:
                reward -= 0.1
                
            # Step penalty
            reward -= 0.01

            # --- Check termination conditions ---
            if self.player.health <= 0:
                self.game_over = True
                self.winner = 'enemy'
                reward -= 100.0
                terminated = True
            elif self.enemy.health <= 0:
                self.game_over = True
                self.winner = 'player'
                reward += 100.0
                terminated = True
            elif self.steps >= self.MAX_STEPS:
                terminated = True
        
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated is always False
            self._get_info(),
        )

    def _handle_player_input(self, movement, space_held):
        # Movement
        accel = pygame.Vector2(0, 0)
        if movement == 1: accel.y = -1  # Up
        if movement == 2: accel.y = 1   # Down
        if movement == 3: accel.x = -1  # Left
        if movement == 4: accel.x = 1   # Right
        if accel.length() > 0:
            accel.scale_to_length(self.TANK_ACCEL)
            self.player.vel += accel
            self.player.rotation = math.atan2(self.player.vel.y, self.player.vel.x)

        # Firing
        if space_held and self.player.fire_cooldown <= 0 and self.player.ammo > 0:
            self.player.fire_cooldown = self.PLAYER_FIRE_COOLDOWN
            self.player.ammo -= 1
            # Sound effect placeholder: # pew!
            proj = self._Projectile(
                pos=self.player.pos + pygame.Vector2(self.TURRET_LENGTH, 0).rotate_rad(self.player.rotation),
                angle=self.player.rotation,
                color=self.COLOR_PLAYER_PROJ
            )
            self.player_projectiles.append(proj)
            
    def _update_enemy_ai(self):
        # Patrol behavior
        self.enemy.pos.y += 2 * self.enemy.patrol_dir
        if self.enemy.pos.y < self.TANK_SIZE or self.enemy.pos.y > self.HEIGHT - self.TANK_SIZE:
            self.enemy.patrol_dir *= -1

        # Aim at player
        direction_to_player = self.player.pos - self.enemy.pos
        self.enemy.rotation = math.atan2(direction_to_player.y, direction_to_player.x)
        
        # Firing
        if self.enemy.fire_cooldown <= 0 and direction_to_player.length() < self.WIDTH / 2:
            self.enemy.fire_cooldown = self.ENEMY_FIRE_COOLDOWN + self.np_random.integers(-10, 10)
            # Sound effect placeholder: # enemy pew!
            proj = self._Projectile(
                pos=self.enemy.pos + pygame.Vector2(self.TURRET_LENGTH, 0).rotate_rad(self.enemy.rotation),
                angle=self.enemy.rotation,
                color=self.COLOR_ENEMY_PROJ
            )
            self.enemy_projectiles.append(proj)

    def _update_tanks(self):
        for tank in [self.player, self.enemy]:
            # Apply friction
            tank.vel *= self.TANK_FRICTION
            if tank.vel.length() < 0.1:
                tank.vel.update(0, 0)

            # Limit speed
            if tank.vel.length() > self.TANK_MAX_SPEED:
                tank.vel.scale_to_length(self.TANK_MAX_SPEED)

            # Update position
            tank.pos += tank.vel

            # Boundary collision
            tank.pos.x = np.clip(tank.pos.x, self.TANK_SIZE / 2, self.WIDTH - self.TANK_SIZE / 2)
            tank.pos.y = np.clip(tank.pos.y, self.TANK_SIZE / 2, self.HEIGHT - self.TANK_SIZE / 2)
            
            # Update cooldowns
            if tank.fire_cooldown > 0:
                tank.fire_cooldown -= 1

    def _update_projectiles(self):
        hit_enemy, was_hit = False, False

        # Player projectiles
        for proj in self.player_projectiles[:]:
            proj.pos += proj.vel
            if proj.pos.distance_to(self.enemy.pos) < self.TANK_SIZE / 2 + self.PROJ_RADIUS:
                self.enemy.health -= 1
                hit_enemy = True
                self._create_explosion(proj.pos, self.COLOR_EXPLOSION)
                self.player_projectiles.remove(proj)
                # Sound effect placeholder: # enemy hit!
            elif not (0 < proj.pos.x < self.WIDTH and 0 < proj.pos.y < self.HEIGHT):
                self._create_explosion(proj.pos, self.COLOR_WALL, small=True)
                self.player_projectiles.remove(proj)

        # Enemy projectiles
        for proj in self.enemy_projectiles[:]:
            proj.pos += proj.vel
            if proj.pos.distance_to(self.player.pos) < self.TANK_SIZE / 2 + self.PROJ_RADIUS:
                self.player.health -= 1
                was_hit = True
                self._create_explosion(proj.pos, self.COLOR_EXPLOSION)
                self.enemy_projectiles.remove(proj)
                # Sound effect placeholder: # player hit!
            elif not (0 < proj.pos.x < self.WIDTH and 0 < proj.pos.y < self.HEIGHT):
                self._create_explosion(proj.pos, self.COLOR_WALL, small=True)
                self.enemy_projectiles.remove(proj)
        
        return hit_enemy, was_hit

    def _create_explosion(self, pos, color, small=False):
        # Sound effect placeholder: # boom!
        num_particles = 10 if small else 25
        max_life = 10 if small else 20
        max_speed = 3 if small else 6
        for _ in range(num_particles):
            self.explosions.append(self._Explosion(pos.copy(), color, max_life, max_speed, self.np_random))

    def _update_explosions(self):
        for exp in self.explosions[:]:
            exp.update()
            if exp.life <= 0:
                self.explosions.remove(exp)

    def _get_observation(self):
        # If reset has not been called, do a dummy render
        if self.player is None:
            self.reset()
        
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw explosions first (background)
        for exp in self.explosions:
            exp.draw(self.screen)
            
        # Draw tanks
        self.player.draw(self.screen, self.TURRET_LENGTH)
        self.enemy.draw(self.screen, self.TURRET_LENGTH)

        # Draw projectiles
        for proj in self.player_projectiles:
            proj.draw(self.screen, self.PROJ_RADIUS)
        for proj in self.enemy_projectiles:
            proj.draw(self.screen, self.PROJ_RADIUS)

    def _render_ui(self):
        # Player Health Bar
        health_pct = self.player.health / self.player.max_health
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_RED, (10, 10, 200, 20))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_GREEN, (10, 10, 200 * health_pct, 20))
        player_health_text = self.font_small.render("PLAYER", True, self.COLOR_TEXT)
        self.screen.blit(player_health_text, (15, 12))

        # Enemy Health Bar
        health_pct = self.enemy.health / self.enemy.max_health
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_RED, (self.WIDTH - 210, 10, 200, 20))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_GREEN, (self.WIDTH - 210 + 200 * (1-health_pct), 10, 200 * health_pct, 20))
        enemy_health_text = self.font_small.render("ENEMY", True, self.COLOR_TEXT)
        self.screen.blit(enemy_health_text, (self.WIDTH - 15 - enemy_health_text.get_width(), 12))

        # Ammo Count
        ammo_text = self.font_large.render(f"AMMO: {self.player.ammo}", True, self.COLOR_TEXT)
        self.screen.blit(ammo_text, (self.WIDTH // 2 - ammo_text.get_width() // 2, self.HEIGHT - 35))
        
        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            self.screen.blit(overlay, (0, 0))
            
            if self.winner == 'player':
                msg = "VICTORY"
                color = self.COLOR_PLAYER
            else:
                msg = "DEFEAT"
                color = self.COLOR_ENEMY
            
            end_text = self.font_large.render(msg, True, color)
            self.screen.blit(end_text, (self.WIDTH // 2 - end_text.get_width() // 2, self.HEIGHT // 2 - end_text.get_height() // 2))


    def _get_info(self):
        # If reset has not been called, return default info
        if self.player is None:
            return {
                "score": 0, "steps": 0, "player_health": self.PLAYER_MAX_HEALTH,
                "enemy_health": self.ENEMY_MAX_HEALTH, "player_ammo": self.PLAYER_MAX_AMMO,
            }
        return {
            "score": self.score,
            "steps": self.steps,
            "player_health": self.player.health,
            "enemy_health": self.enemy.health,
            "player_ammo": self.player.ammo,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        obs, _ = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert obs.dtype == np.uint8
        
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

    # --- Helper Classes ---
    class _Tank:
        def __init__(self, pos, color, max_health, max_ammo=0):
            self.pos = pos
            self.vel = pygame.Vector2(0, 0)
            self.color = color
            self.rotation = 0.0  # Radians
            self.health = max_health
            self.max_health = max_health
            self.ammo = max_ammo
            self.max_ammo = max_ammo
            self.fire_cooldown = 0
            
        def draw(self, surface, turret_length):
            size = GameEnv.TANK_SIZE
            rect = pygame.Rect(self.pos.x - size/2, self.pos.y - size/2, size, size)
            
            # Body
            pygame.draw.rect(surface, self.color, rect, border_radius=3)
            # FIX: Ensure color components are integers and within the valid [0, 255] range.
            darker_body_color = tuple(int(c * 0.7) for c in self.color)
            pygame.draw.rect(surface, darker_body_color, rect, width=2, border_radius=3)

            # Turret
            end_pos = self.pos + pygame.Vector2(turret_length, 0).rotate_rad(self.rotation)
            darker_turret_color = tuple(int(c * 0.8) for c in self.color)
            brighter_turret_color = tuple(min(255, int(c * 1.2)) for c in self.color)
            pygame.draw.line(surface, darker_turret_color, self.pos, end_pos, width=6)
            pygame.draw.line(surface, brighter_turret_color, self.pos, end_pos, width=2)

    class _Projectile:
        def __init__(self, pos, angle, color):
            self.pos = pos
            self.vel = pygame.Vector2(GameEnv.PROJ_SPEED, 0).rotate_rad(angle)
            self.color = color
        
        def draw(self, surface, radius):
            x, y = int(self.pos.x), int(self.pos.y)
            # Draw a filled, anti-aliased circle with a bright core
            pygame.gfxdraw.filled_circle(surface, x, y, radius, self.color)
            pygame.gfxdraw.aacircle(surface, x, y, radius, self.color)
            core_color = (255, 255, 255)
            pygame.gfxdraw.filled_circle(surface, x, y, radius // 2, core_color)
            
    class _Explosion:
        def __init__(self, pos, color, max_life, max_speed, np_random):
            self.pos = pos
            # Use np_random for reproducibility
            vx = np_random.uniform(-max_speed, max_speed)
            vy = np_random.uniform(-max_speed, max_speed)
            self.vel = pygame.Vector2(vx, vy)
            self.color = color
            self.life = max_life
            self.max_life = max_life
        
        def update(self):
            self.pos += self.vel
            self.vel *= 0.85 # friction
            self.life -= 1
            
        def draw(self, surface):
            if self.life > 0:
                alpha = int(255 * (self.life / self.max_life))
                radius = int(8 * (1 - self.life / self.max_life))
                if radius * 2 <= 0: return # Avoid creating 0-size surface
                temp_surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
                color_with_alpha = self.color + (alpha,)
                pygame.gfxdraw.filled_circle(temp_surf, radius, radius, radius, color_with_alpha)
                surface.blit(temp_surf, (int(self.pos.x - radius), int(self.pos.y - radius)))

if __name__ == '__main__':
    # This block allows you to run the file directly to play the game
    import sys
    
    # Re-enable video driver for human play
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv()
    obs, info = env.reset(seed=42)
    
    # --- Pygame setup for human play ---
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption(GameEnv.game_description)
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    print(GameEnv.user_guide)

    while running:
        # --- Event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        # --- Action mapping from keyboard ---
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
        total_reward += reward

        if terminated or truncated:
            print(f"Episode finished. Total reward: {total_reward:.2f}, Final Info: {info}")
            obs, info = env.reset()
            total_reward = 0
            # A small pause on game over
            pygame.time.wait(1500)

        # --- Rendering ---
        # The observation is already a rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        # --- Frame rate control ---
        clock.tick(GameEnv.FPS)

    env.close()
    pygame.quit()
    sys.exit()