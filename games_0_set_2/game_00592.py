
# Generated: 2025-08-27T14:06:55.504689
# Source Brief: brief_00592.md
# Brief Index: 592

        
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

    user_guide = (
        "Controls: Arrow keys to move. Space to shoot in your last direction of movement."
    )

    game_description = (
        "Survive a relentless zombie horde in a top-down arena shooter. "
        "Position yourself carefully and eliminate all the undead to win."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.PLAYER_SIZE = 20
        self.PLAYER_SPEED = 4
        self.PLAYER_HEALTH_MAX = 100
        self.ZOMBIE_COUNT = 25
        self.ZOMBIE_SIZE = 20
        self.ZOMBIE_SPEED = 1.2
        self.ZOMBIE_HEALTH_MAX = 10
        self.ZOMBIE_DAMAGE = 10
        self.PROJECTILE_WIDTH = 12
        self.PROJECTILE_HEIGHT = 4
        self.PROJECTILE_SPEED = 10
        self.PROJECTILE_DAMAGE = 10
        self.WEAPON_COOLDOWN_FRAMES = 8  # Cooldown in frames (30fps)
        self.MAX_STEPS = 1000
        self.ARENA_MARGIN = 15

        # Colors
        self.COLOR_BG = (25, 20, 20)
        self.COLOR_WALL = (60, 60, 70)
        self.COLOR_PLAYER = (50, 255, 150)
        self.COLOR_PLAYER_GLOW = (50, 255, 150, 50)
        self.COLOR_ZOMBIE = (220, 50, 50)
        self.COLOR_PROJECTILE = (255, 255, 0)
        self.COLOR_WHITE = (255, 255, 255)
        self.COLOR_HEALTH_BAR = (50, 200, 50)
        self.COLOR_HEALTH_BAR_BG = (100, 40, 40)
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.COLOR_END_TEXT = (255, 255, 255)

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
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_end = pygame.font.SysFont("monospace", 50, bold=True)

        # Initialize state variables
        self.player_pos = None
        self.player_health = None
        self.last_move_direction = None
        self.zombies = []
        self.projectiles = []
        self.particles = []
        self.weapon_cooldown = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # This will be initialized in reset()
        self.np_random = None

        self.reset()
        
        # Self-validation check
        # self.validate_implementation()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.player_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
        self.player_health = self.PLAYER_HEALTH_MAX
        self.last_move_direction = pygame.Vector2(0, -1)  # Default up

        self.zombies = []
        self._spawn_zombies(self.ZOMBIE_COUNT)

        self.projectiles = []
        self.particles = []
        self.weapon_cooldown = 0
        self.steps = 0
        self.score = 0
        self.game_over = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            # If game is over, no actions have effect, just return current state
            reward = 0
            terminated = True
            return self._get_observation(), reward, terminated, False, self._get_info()

        # --- Update game state ---
        self.steps += 1
        reward = 0.01  # Small reward for surviving a step

        self._handle_input(action)
        self._update_entities()
        reward += self._handle_collisions()
        
        # --- Update score and check for termination ---
        self.score += reward
        terminated = self._check_termination()

        if terminated and not self.game_over:
            self.game_over = True
            if self.player_health <= 0:
                reward -= 100  # Large penalty for dying
            elif not self.zombies:
                reward += 100  # Large reward for winning

        # Cap rewards to specified range
        reward = np.clip(reward, -100, 100)

        return (
            self._get_observation(),
            float(reward),
            terminated,
            False,
            self._get_info(),
        )

    def _spawn_zombies(self, count):
        for _ in range(count):
            while True:
                pos = pygame.Vector2(
                    self.np_random.uniform(self.ARENA_MARGIN, self.WIDTH - self.ARENA_MARGIN),
                    self.np_random.uniform(self.ARENA_MARGIN, self.HEIGHT - self.ARENA_MARGIN),
                )
                if pos.distance_to(self.player_pos) > 150:  # Spawn away from player
                    self.zombies.append(
                        {
                            "pos": pos,
                            "health": self.ZOMBIE_HEALTH_MAX,
                            "hit_flash": 0,
                        }
                    )
                    break
    
    def _handle_input(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1

        move_vector = pygame.Vector2(0, 0)
        if movement == 1: move_vector.y = -1  # Up
        elif movement == 2: move_vector.y = 1  # Down
        elif movement == 3: move_vector.x = -1  # Left
        elif movement == 4: move_vector.x = 1  # Right
        
        if move_vector.length() > 0:
            move_vector.normalize_ip()
            self.last_move_direction = move_vector.copy()
        
        self.player_pos += move_vector * self.PLAYER_SPEED
        self.player_pos.x = np.clip(self.player_pos.x, self.ARENA_MARGIN, self.WIDTH - self.ARENA_MARGIN)
        self.player_pos.y = np.clip(self.player_pos.y, self.ARENA_MARGIN, self.HEIGHT - self.ARENA_MARGIN)

        if space_held and self.weapon_cooldown == 0:
            self._fire_projectile()

    def _fire_projectile(self):
        # sfx: player_shoot.wav
        self.weapon_cooldown = self.WEAPON_COOLDOWN_FRAMES
        
        # Create a projectile slightly in front of the player
        start_pos = self.player_pos + self.last_move_direction * (self.PLAYER_SIZE / 2)
        
        self.projectiles.append(
            {"pos": start_pos, "vel": self.last_move_direction.copy()}
        )
        # Muzzle flash effect
        self._create_particles(start_pos, 5, self.COLOR_PROJECTILE, 1, 5, 4)

    def _update_entities(self):
        # Update weapon cooldown
        if self.weapon_cooldown > 0:
            self.weapon_cooldown -= 1

        # Update projectiles
        for p in self.projectiles[:]:
            p["pos"] += p["vel"] * self.PROJECTILE_SPEED
            if not (0 < p["pos"].x < self.WIDTH and 0 < p["pos"].y < self.HEIGHT):
                self.projectiles.remove(p)

        # Update zombies
        for z in self.zombies:
            direction = (self.player_pos - z["pos"]).normalize() if (self.player_pos - z["pos"]).length() > 0 else pygame.Vector2(0,0)
            z["pos"] += direction * self.ZOMBIE_SPEED
            if z["hit_flash"] > 0:
                z["hit_flash"] -= 1

        # Update particles
        for particle in self.particles[:]:
            particle["pos"] += particle["vel"]
            particle["lifetime"] -= 1
            if particle["lifetime"] <= 0:
                self.particles.remove(particle)

    def _handle_collisions(self):
        reward = 0
        player_rect = pygame.Rect(self.player_pos.x - self.PLAYER_SIZE / 2, self.player_pos.y - self.PLAYER_SIZE / 2, self.PLAYER_SIZE, self.PLAYER_SIZE)

        # Projectiles vs Zombies
        for p in self.projectiles[:]:
            proj_rect = pygame.Rect(p["pos"].x, p["pos"].y, self.PROJECTILE_WIDTH, self.PROJECTILE_HEIGHT)
            if p['vel'].x != 0: # Horizontal
                proj_rect.height = self.PROJECTILE_HEIGHT
                proj_rect.width = self.PROJECTILE_WIDTH
            else: # Vertical
                proj_rect.height = self.PROJECTILE_WIDTH
                proj_rect.width = self.PROJECTILE_HEIGHT
            proj_rect.center = p['pos']


            for z in self.zombies[:]:
                zombie_rect = pygame.Rect(z["pos"].x - self.ZOMBIE_SIZE / 2, z["pos"].y - self.ZOMBIE_SIZE / 2, self.ZOMBIE_SIZE, self.ZOMBIE_SIZE)
                if zombie_rect.colliderect(proj_rect):
                    # sfx: zombie_hit.wav
                    z["health"] -= self.PROJECTILE_DAMAGE
                    z["hit_flash"] = 3 # Flash for 3 frames
                    self._create_particles(z["pos"], 10, self.COLOR_ZOMBIE, 2, 8, 2)
                    if p in self.projectiles: self.projectiles.remove(p)
                    
                    if z["health"] <= 0:
                        # sfx: zombie_die.wav
                        reward += 1.0
                        self._create_particles(z["pos"], 30, self.COLOR_ZOMBIE, 3, 15, 3)
                        self.zombies.remove(z)
                    break 

        # Zombies vs Player
        for z in self.zombies:
            zombie_rect = pygame.Rect(z["pos"].x - self.ZOMBIE_SIZE / 2, z["pos"].y - self.ZOMBIE_SIZE / 2, self.ZOMBIE_SIZE, self.ZOMBIE_SIZE)
            if player_rect.colliderect(zombie_rect):
                # sfx: player_hurt.wav
                self.player_health -= self.ZOMBIE_DAMAGE
                self.player_health = max(0, self.player_health)
                # Knockback effect
                knockback = (self.player_pos - z["pos"]).normalize() * 5
                self.player_pos += knockback
                self._create_particles(self.player_pos, 15, self.COLOR_PLAYER, 2, 10, 2)
                
                # Prevent instant multi-hits by moving zombie back
                z['pos'] -= knockback * 2

        return reward

    def _check_termination(self):
        return (
            self.player_health <= 0
            or not self.zombies
            or self.steps >= self.MAX_STEPS
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw arena walls
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, 0, self.WIDTH, self.HEIGHT), self.ARENA_MARGIN)

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p["lifetime"] / p["start_lifetime"]))
            p['color'] = p['base_color'] + (alpha,) # Add alpha to color tuple
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), int(p['size']), p['color'])

        # Draw projectiles
        for p in self.projectiles:
            angle = math.degrees(math.atan2(-p["vel"].y, p["vel"].x))
            if p['vel'].x != 0:
                surf = pygame.Surface((self.PROJECTILE_WIDTH, self.PROJECTILE_HEIGHT), pygame.SRCALPHA)
            else:
                surf = pygame.Surface((self.PROJECTILE_HEIGHT, self.PROJECTILE_WIDTH), pygame.SRCALPHA)
            surf.fill(self.COLOR_PROJECTILE)
            rotated_surf = pygame.transform.rotate(surf, angle)
            rect = rotated_surf.get_rect(center=p["pos"])
            self.screen.blit(rotated_surf, rect)

        # Draw zombies
        for z in self.zombies:
            rect = pygame.Rect(0, 0, self.ZOMBIE_SIZE, self.ZOMBIE_SIZE)
            rect.center = z["pos"]
            color = self.COLOR_WHITE if z["hit_flash"] > 0 else self.COLOR_ZOMBIE
            pygame.draw.rect(self.screen, color, rect)
            pygame.draw.rect(self.screen, self.COLOR_BG, rect, 1)

        # Draw player
        player_rect = pygame.Rect(0, 0, self.PLAYER_SIZE, self.PLAYER_SIZE)
        player_rect.center = self.player_pos
        pygame.gfxdraw.box(self.screen, player_rect, self.COLOR_PLAYER_GLOW)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect.inflate(-4, -4))
        
        # Draw aiming direction indicator
        aim_end = self.player_pos + self.last_move_direction * (self.PLAYER_SIZE)
        pygame.draw.line(self.screen, self.COLOR_PLAYER, self.player_pos, aim_end, 2)


    def _render_ui(self):
        # Health bar
        health_ratio = self.player_health / self.PLAYER_HEALTH_MAX
        bar_width = 200
        bar_height = 20
        health_bar_rect = pygame.Rect(10, 10, bar_width, bar_height)
        current_health_rect = pygame.Rect(10, 10, int(bar_width * health_ratio), bar_height)
        
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, health_bar_rect)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, current_health_rect)
        pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, health_bar_rect, 1)
        
        health_text = self.font_ui.render(f"HP: {self.player_health}", True, self.COLOR_UI_TEXT)
        self.screen.blit(health_text, (15, 12))

        # Zombie count
        zombie_text = self.font_ui.render(f"Zombies: {len(self.zombies)}", True, self.COLOR_UI_TEXT)
        text_rect = zombie_text.get_rect(topright=(self.WIDTH - 15, 12))
        self.screen.blit(zombie_text, text_rect)

        # Game Over / Win message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            if self.player_health <= 0:
                end_text_str = "GAME OVER"
            elif not self.zombies:
                end_text_str = "YOU WIN!"
            else: # Time out
                end_text_str = "TIME UP"
                
            end_text = self.font_end.render(end_text_str, True, self.COLOR_END_TEXT)
            text_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, text_rect)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_health": self.player_health,
            "zombies_remaining": len(self.zombies),
        }

    def _create_particles(self, pos, count, color, speed, lifetime, size):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed * self.np_random.uniform(0.5, 1.5)
            self.particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "base_color": color,
                "color": color,
                "lifetime": self.np_random.integers(lifetime // 2, lifetime),
                "start_lifetime": lifetime,
                "size": self.np_random.uniform(size // 2, size)
            })

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation:
        """
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
        assert isinstance(reward, float)
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Zombie Arena")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        movement = 0  # 0=none
        space_held = 0 # 0=released
        shift_held = 0 # 0=released

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1

        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Convert observation back to a surface for display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Game Over! Final Score: {info['score']:.2f} in {info['steps']} steps.")
            pygame.time.wait(3000) # Pause for 3 seconds
            obs, info = env.reset()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        clock.tick(30) # Run at 30 FPS

    env.close()