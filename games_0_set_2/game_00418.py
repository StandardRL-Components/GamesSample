
# Generated: 2025-08-27T13:35:27.356102
# Source Brief: brief_00418.md
# Brief Index: 418

        
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


class Explosion:
    """A simple class for explosion particle effects."""
    def __init__(self, pos, max_radius=30, life=15, color_start=(255, 165, 0), color_end=(255, 255, 0)):
        self.pos = pos
        self.max_radius = max_radius
        self.life = life
        self.max_life = life
        self.color_start = color_start
        self.color_end = color_end

    def update(self):
        self.life -= 1

    def draw(self, surface):
        if self.life > 0:
            progress = self.life / self.max_life
            current_radius = int(self.max_radius * (1 - progress))
            alpha = int(255 * progress)
            
            # Interpolate color from orange to yellow
            r = int(self.color_start[0] + (self.color_end[0] - self.color_start[0]) * (1 - progress))
            g = int(self.color_start[1] + (self.color_end[1] - self.color_start[1]) * (1 - progress))
            b = int(self.color_start[2] + (self.color_end[2] - self.color_start[2]) * (1 - progress))
            
            # Draw a filled circle with alpha
            temp_surf = pygame.Surface((current_radius*2, current_radius*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, (r, g, b, alpha), (current_radius, current_radius), current_radius)
            surface.blit(temp_surf, (int(self.pos[0] - current_radius), int(self.pos[1] - current_radius)))

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ↑/↓ to move forward/backward, ←/→ to turn. Press space to fire."
    )

    game_description = (
        "Control a tank in a top-down arena. Destroy the enemy tank before it destroys you."
    )

    auto_advance = True

    # --- Constants ---
    # Screen
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Colors
    COLOR_BG = (40, 40, 45)
    COLOR_WALL = (80, 80, 90)
    COLOR_PLAYER = (0, 200, 100)
    COLOR_PLAYER_GLOW = (0, 200, 100, 50)
    COLOR_ENEMY = (255, 50, 50)
    COLOR_PROJECTILE = (255, 255, 100)
    COLOR_UI_TEXT = (230, 230, 230)
    COLOR_UI_BG = (60, 60, 70)
    COLOR_UI_HEALTH = (0, 255, 120)
    COLOR_UI_HEALTH_ENEMY = (255, 80, 80)
    
    # Game parameters
    MAX_STEPS = 1500
    WALL_THICKNESS = 15
    
    # Tank parameters
    TANK_RADIUS = 12
    TANK_SPEED = 2.0
    TANK_ROTATION_SPEED = 0.07  # radians per frame
    TURRET_LENGTH = 20
    TURRET_WIDTH = 6
    MAX_HEALTH = 100
    MAX_AMMO = 50
    FIRE_COOLDOWN = 10 # frames
    
    # Enemy parameters
    ENEMY_FIRE_INTERVAL = 25 # frames
    
    # Projectile parameters
    PROJECTILE_SPEED = 6.0
    PROJECTILE_RADIUS = 3
    PROJECTILE_DAMAGE = 10

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 48, bold=True)
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        self.player_pos = np.array([100.0, self.SCREEN_HEIGHT / 2.0])
        self.player_angle = 0.0
        self.player_health = self.MAX_HEALTH
        self.player_ammo = self.MAX_AMMO
        self.player_fire_cooldown = 0
        
        self.enemy_pos = np.array([self.SCREEN_WIDTH - 100.0, self.SCREEN_HEIGHT / 2.0])
        self.enemy_angle = math.pi
        self.enemy_health = self.MAX_HEALTH
        self.enemy_fire_cooldown = self.ENEMY_FIRE_INTERVAL
        self.enemy_patrol_dir = 1

        self.projectiles = []
        self.explosions = []
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        step_reward = 0
        
        self._handle_player_action(action)
        self._update_enemy_ai()
        step_reward += self._update_projectiles()
        self._update_explosions()

        self.steps += 1
        
        terminated = self.player_health <= 0 or self.enemy_health <= 0 or self.steps >= self.MAX_STEPS
        if terminated:
            self.game_over = True
            if self.enemy_health <= 0 and self.player_health > 0:
                step_reward += 100
                self.win = True
            elif self.player_health <= 0:
                step_reward -= 100

        self.score += step_reward
        
        return (
            self._get_observation(),
            step_reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_player_action(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        # Cooldowns
        if self.player_fire_cooldown > 0:
            self.player_fire_cooldown -= 1

        # Rotation
        if movement == 3: # Left
            self.player_angle -= self.TANK_ROTATION_SPEED
        if movement == 4: # Right
            self.player_angle += self.TANK_ROTATION_SPEED
            
        # Movement
        move_vec = np.array([math.cos(self.player_angle), math.sin(self.player_angle)])
        if movement == 1: # Up
            self.player_pos += move_vec * self.TANK_SPEED
        if movement == 2: # Down
            self.player_pos -= move_vec * self.TANK_SPEED
        
        # Firing
        if space_held and self.player_fire_cooldown == 0 and self.player_ammo > 0:
            # SFX: Player shoot
            turret_end = self.player_pos + move_vec * self.TURRET_LENGTH
            self.projectiles.append({
                "pos": turret_end,
                "angle": self.player_angle,
                "owner": "player",
                "color": self.COLOR_PROJECTILE
            })
            self.player_ammo -= 1
            self.player_fire_cooldown = self.FIRE_COOLDOWN

        # Clamp position
        self.player_pos[0] = np.clip(self.player_pos[0], self.WALL_THICKNESS + self.TANK_RADIUS, self.SCREEN_WIDTH - self.WALL_THICKNESS - self.TANK_RADIUS)
        self.player_pos[1] = np.clip(self.player_pos[1], self.WALL_THICKNESS + self.TANK_RADIUS, self.SCREEN_HEIGHT - self.WALL_THICKNESS - self.TANK_RADIUS)

    def _update_enemy_ai(self):
        # Movement
        self.enemy_pos[0] += self.TANK_SPEED * 0.5 * self.enemy_patrol_dir
        if self.enemy_pos[0] <= self.WALL_THICKNESS + self.TANK_RADIUS or \
           self.enemy_pos[0] >= self.SCREEN_WIDTH - self.WALL_THICKNESS - self.TANK_RADIUS:
            self.enemy_patrol_dir *= -1
            
        # Aiming
        dx = self.player_pos[0] - self.enemy_pos[0]
        dy = self.player_pos[1] - self.enemy_pos[1]
        self.enemy_angle = math.atan2(dy, dx)
        
        # Firing
        self.enemy_fire_cooldown -= 1
        if self.enemy_fire_cooldown <= 0:
            # SFX: Enemy shoot
            move_vec = np.array([math.cos(self.enemy_angle), math.sin(self.enemy_angle)])
            turret_end = self.enemy_pos + move_vec * self.TURRET_LENGTH
            self.projectiles.append({
                "pos": turret_end,
                "angle": self.enemy_angle,
                "owner": "enemy",
                "color": self.COLOR_ENEMY
            })
            self.enemy_fire_cooldown = self.ENEMY_FIRE_INTERVAL

    def _update_projectiles(self):
        reward = 0
        projectiles_to_keep = []
        for proj in self.projectiles:
            move_vec = np.array([math.cos(proj["angle"]), math.sin(proj["angle"])])
            proj["pos"] += move_vec * self.PROJECTILE_SPEED
            
            # Check wall collisions
            if not (self.WALL_THICKNESS < proj["pos"][0] < self.SCREEN_WIDTH - self.WALL_THICKNESS and \
                    self.WALL_THICKNESS < proj["pos"][1] < self.SCREEN_HEIGHT - self.WALL_THICKNESS):
                # SFX: Projectile hit wall
                self.explosions.append(Explosion(proj["pos"], max_radius=10, life=8))
                if proj["owner"] == "player":
                    reward -= 0.01 # Missed shot penalty
                continue

            # Check tank collisions
            hit = False
            if proj["owner"] == "player":
                dist = np.linalg.norm(proj["pos"] - self.enemy_pos)
                if dist < self.TANK_RADIUS + self.PROJECTILE_RADIUS:
                    self.enemy_health -= self.PROJECTILE_DAMAGE
                    reward += 10.0 # Reward for hitting enemy
                    hit = True
            elif proj["owner"] == "enemy":
                dist = np.linalg.norm(proj["pos"] - self.player_pos)
                if dist < self.TANK_RADIUS + self.PROJECTILE_RADIUS:
                    self.player_health -= self.PROJECTILE_DAMAGE
                    # Negative reward for getting hit is implicitly handled by losing
                    hit = True

            if hit:
                # SFX: Tank hit explosion
                self.explosions.append(Explosion(proj["pos"]))
            else:
                projectiles_to_keep.append(proj)
        
        self.projectiles = projectiles_to_keep
        return reward

    def _update_explosions(self):
        for exp in self.explosions:
            exp.update()
        self.explosions = [exp for exp in self.explosions if exp.life > 0]

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
            "player_health": self.player_health,
            "enemy_health": self.enemy_health,
            "player_ammo": self.player_ammo
        }

    def _render_game(self):
        # Walls
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, 0, self.SCREEN_WIDTH, self.WALL_THICKNESS))
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, self.SCREEN_HEIGHT - self.WALL_THICKNESS, self.SCREEN_WIDTH, self.WALL_THICKNESS))
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, 0, self.WALL_THICKNESS, self.SCREEN_HEIGHT))
        pygame.draw.rect(self.screen, self.COLOR_WALL, (self.SCREEN_WIDTH - self.WALL_THICKNESS, 0, self.WALL_THICKNESS, self.SCREEN_HEIGHT))

        # Explosions
        for exp in self.explosions:
            exp.draw(self.screen)

        # Projectiles
        for proj in self.projectiles:
            pygame.gfxdraw.filled_circle(self.screen, int(proj["pos"][0]), int(proj["pos"][1]), self.PROJECTILE_RADIUS, proj["color"])
            pygame.gfxdraw.aacircle(self.screen, int(proj["pos"][0]), int(proj["pos"][1]), self.PROJECTILE_RADIUS, proj["color"])

        # Tanks
        self._render_tank(self.enemy_pos, self.enemy_angle, self.COLOR_ENEMY, self.COLOR_ENEMY)
        self._render_tank(self.player_pos, self.player_angle, self.COLOR_PLAYER, self.COLOR_PLAYER_GLOW)
        
        if self.game_over:
            msg = "YOU WIN" if self.win else "GAME OVER"
            color = self.COLOR_PLAYER if self.win else self.COLOR_ENEMY
            text_surf = self.font_game_over.render(msg, True, color)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(text_surf, text_rect)


    def _render_tank(self, pos, angle, color, glow_color):
        x, y = int(pos[0]), int(pos[1])
        
        # Glow effect
        glow_radius = int(self.TANK_RADIUS * 1.5)
        temp_surf = pygame.Surface((glow_radius*2, glow_radius*2), pygame.SRCALPHA)
        pygame.draw.circle(temp_surf, glow_color, (glow_radius, glow_radius), glow_radius)
        self.screen.blit(temp_surf, (x - glow_radius, y - glow_radius))

        # Turret
        turret_end_x = x + self.TURRET_LENGTH * math.cos(angle)
        turret_end_y = y + self.TURRET_LENGTH * math.sin(angle)
        pygame.draw.line(self.screen, color, (x, y), (turret_end_x, turret_end_y), self.TURRET_WIDTH)

        # Body
        pygame.gfxdraw.filled_circle(self.screen, x, y, self.TANK_RADIUS, color)
        pygame.gfxdraw.aacircle(self.screen, x, y, self.TANK_RADIUS, color)

    def _render_ui(self):
        # Player UI
        health_pct = max(0, self.player_health / self.MAX_HEALTH)
        health_bar_width = int(150 * health_pct)
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, (10, 10, 150, 20))
        pygame.draw.rect(self.screen, self.COLOR_UI_HEALTH, (10, 10, health_bar_width, 20))
        
        ammo_text = self.font_ui.render(f"AMMO: {self.player_ammo}", True, self.COLOR_UI_TEXT)
        self.screen.blit(ammo_text, (10, 35))

        # Enemy UI
        enemy_health_pct = max(0, self.enemy_health / self.MAX_HEALTH)
        enemy_health_bar_width = int(150 * enemy_health_pct)
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, (self.SCREEN_WIDTH - 160, 10, 150, 20))
        pygame.draw.rect(self.screen, self.COLOR_UI_HEALTH_ENEMY, (self.SCREEN_WIDTH - 160, 10, enemy_health_bar_width, 20))

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
        assert trunc == False
        assert isinstance(info, dict)
        
        # Test game mechanics assertions
        self.reset()
        assert self.player_health == self.MAX_HEALTH
        assert self.player_ammo == self.MAX_AMMO
        
        fire_action = [0, 1, 0] # no-op move, fire, no-op shift
        for _ in range(5):
            self.step(fire_action)
        
        # After 5 no-op steps with firing, check ammo. Cooldown is 10, so only 1 shot fired
        assert self.player_ammo == self.MAX_AMMO - 1
        
        # Test projectile damage
        self.reset()
        initial_enemy_health = self.enemy_health
        self.projectiles.append({
            "pos": self.enemy_pos.copy(),
            "angle": 0, "owner": "player", "color": self.COLOR_PROJECTILE
        })
        self._update_projectiles()
        assert self.enemy_health == initial_enemy_health - self.PROJECTILE_DAMAGE
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Set render to True to see the game window
    render = True
    if render:
        render_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        pygame.display.set_caption("Tank Arena")
        
    total_reward = 0
    
    # Game loop
    while not done:
        # Get user input
        keys = pygame.key.get_pressed()
        
        movement = 0 # no-op
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
            
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        if render:
            # Transpose obs for pygame display (H, W, C) -> (W, H, C)
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            render_screen.blit(surf, (0, 0))
            pygame.display.flip()
        
        # Handle closing the window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
                
        env.clock.tick(30) # 30 FPS

    print(f"Game Over! Final Score: {info['score']}")
    pygame.quit()