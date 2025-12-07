
# Generated: 2025-08-27T18:36:02.171518
# Source Brief: brief_01882.md
# Brief Index: 1882

        
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
        "Controls: Arrow keys to move. Space to shoot in your last moved direction."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Hunt down swarming red bugs in a frantic top-down arena shooter. Eliminate 25 to win, but lose 3 lives and you're done. Difficulty ramps up as you progress!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.ARENA_MARGIN = 20

        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_ARENA_BORDER = (200, 200, 220)
        self.COLOR_PLAYER = (50, 255, 150)
        self.COLOR_PLAYER_GUN = (200, 255, 220)
        self.COLOR_BUG = (255, 80, 80)
        self.COLOR_PROJECTILE = (255, 255, 100)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_HEALTH_BAR = (100, 220, 100)
        self.COLOR_HEALTH_BAR_BG = (120, 50, 50)
        self.COLOR_SCORE = (255, 200, 0)

        # Player settings
        self.PLAYER_SIZE = 12
        self.PLAYER_SPEED = 4.0
        self.PLAYER_MAX_HEALTH = 3
        self.PLAYER_INVINCIBILITY_FRAMES = 60 # 2 seconds

        # Bug settings
        self.BUG_SIZE = 10
        self.BUG_INITIAL_SPEED = 1.0
        self.BUG_SPEED_INCREMENT = 0.2
        self.BUG_MAX_SPEED = 4.0
        self.INITIAL_SPAWN_INTERVAL = 90 # 3 seconds
        self.SPAWN_INTERVAL_DECREMENT = 10
        self.MIN_SPAWN_INTERVAL = 30 # 1 second

        # Projectile settings
        self.PROJECTILE_SIZE = 4
        self.PROJECTILE_SPEED = 8.0
        self.PROJECTILE_COOLDOWN_FRAMES = 8 # ~4 shots/sec

        # Game rules
        self.WIN_CONDITION_KILLS = 25
        self.MAX_STEPS = 30 * 60 # 60 seconds

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_game_over = pygame.font.Font(None, 72)

        # --- Internal State ---
        # Note: All mutable state variables are initialized in reset()
        self.player_pos = None
        self.player_health = None
        self.player_aim_direction = None
        self.player_invincibility_timer = None
        self.projectile_cooldown = None
        self.last_space_held = None
        self.bugs = None
        self.projectiles = None
        self.particles = None
        self.bug_spawn_timer = None
        self.current_spawn_interval = None
        self.current_bug_speed = None
        self.bugs_killed_total = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.game_won = None

        self.reset()
        
        # Self-check to ensure API compliance
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.player_pos = pygame.math.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
        self.player_health = self.PLAYER_MAX_HEALTH
        self.player_aim_direction = pygame.math.Vector2(0, -1)  # Start aiming up
        self.player_invincibility_timer = 0
        self.projectile_cooldown = 0
        self.last_space_held = False

        self.bugs = []
        self.projectiles = []
        self.particles = []

        self.current_spawn_interval = self.INITIAL_SPAWN_INTERVAL
        self.current_bug_speed = self.BUG_INITIAL_SPEED
        self.bug_spawn_timer = self.current_spawn_interval

        self.bugs_killed_total = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.game_won = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = -0.001  # Small penalty for existing
        self.steps += 1

        if not self.game_over:
            self._handle_input(action)
            self._update_game_state()
            reward += self._handle_collisions()
            self._update_difficulty()

        terminated = self._check_termination()
        if terminated:
            if self.game_won:
                reward += 100
            else:
                reward -= 100 # Penalty for losing all lives or running out of time

        if self.auto_advance:
            self.clock.tick(self.FPS)

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _handle_input(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1

        # --- Movement and Aiming ---
        move_vec = pygame.math.Vector2(0, 0)
        if movement == 1: move_vec.y = -1
        elif movement == 2: move_vec.y = 1
        elif movement == 3: move_vec.x = -1
        elif movement == 4: move_vec.x = 1

        if move_vec.length() > 0:
            move_vec.normalize_ip()
            self.player_pos += move_vec * self.PLAYER_SPEED
            self.player_aim_direction = move_vec.copy()

        # Clamp player position to arena
        self.player_pos.x = np.clip(self.player_pos.x, self.ARENA_MARGIN + self.PLAYER_SIZE, self.WIDTH - self.ARENA_MARGIN - self.PLAYER_SIZE)
        self.player_pos.y = np.clip(self.player_pos.y, self.ARENA_MARGIN + self.PLAYER_SIZE, self.HEIGHT - self.ARENA_MARGIN - self.PLAYER_SIZE)

        # --- Shooting ---
        if space_held and not self.last_space_held and self.projectile_cooldown <= 0:
            # sound: player_shoot.wav
            self.projectiles.append({
                'pos': self.player_pos.copy(),
                'vel': self.player_aim_direction * self.PROJECTILE_SPEED
            })
            self.projectile_cooldown = self.PROJECTILE_COOLDOWN_FRAMES
            # Muzzle flash particle effect
            for _ in range(5):
                self._create_particle(self.player_pos + self.player_aim_direction * 15, self.COLOR_PROJECTILE, 10, 2.0)


        self.last_space_held = space_held

    def _update_game_state(self):
        # Update timers
        if self.projectile_cooldown > 0: self.projectile_cooldown -= 1
        if self.player_invincibility_timer > 0: self.player_invincibility_timer -= 1

        # Move projectiles and remove off-screen ones
        self.projectiles = [p for p in self.projectiles if self.ARENA_MARGIN < p['pos'].x < self.WIDTH - self.ARENA_MARGIN and self.ARENA_MARGIN < p['pos'].y < self.HEIGHT - self.ARENA_MARGIN]
        for p in self.projectiles:
            p['pos'] += p['vel']

        # Move bugs
        for bug in self.bugs:
            direction_to_player = (self.player_pos - bug['pos'])
            if direction_to_player.length() > 0:
                direction_to_player.normalize_ip()
            bug['pos'] += direction_to_player * self.current_bug_speed

        # Update particles
        self.particles = [p for p in self.particles if p['lifespan'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifespan'] -= 1

        # Spawn new bugs
        self.bug_spawn_timer -= 1
        if self.bug_spawn_timer <= 0:
            # sound: bug_spawn.wav
            side = self.np_random.integers(4)
            if side == 0: x, y = self.np_random.uniform(0, self.WIDTH), self.ARENA_MARGIN / 2
            elif side == 1: x, y = self.np_random.uniform(0, self.WIDTH), self.HEIGHT - self.ARENA_MARGIN / 2
            elif side == 2: x, y = self.ARENA_MARGIN / 2, self.np_random.uniform(0, self.HEIGHT)
            else: x, y = self.WIDTH - self.ARENA_MARGIN / 2, self.np_random.uniform(0, self.HEIGHT)
            
            self.bugs.append({'pos': pygame.math.Vector2(x, y)})
            self.bug_spawn_timer = self.current_spawn_interval

    def _handle_collisions(self):
        reward = 0
        
        # Projectile-bug collisions
        projectiles_to_remove = []
        bugs_to_remove = []
        for i, p in enumerate(self.projectiles):
            for j, bug in enumerate(self.bugs):
                if j in bugs_to_remove: continue
                if p['pos'].distance_to(bug['pos']) < self.BUG_SIZE + self.PROJECTILE_SIZE:
                    # sound: bug_die.wav
                    bugs_to_remove.append(j)
                    if i not in projectiles_to_remove: projectiles_to_remove.append(i)
                    self.score += 10
                    self.bugs_killed_total += 1
                    reward += 1.0 # Reward for killing a bug
                    for _ in range(20): # Explosion effect
                        self._create_particle(bug['pos'], self.COLOR_BUG, 20, 3.0)
                    break
        
        self.projectiles = [p for i, p in enumerate(self.projectiles) if i not in projectiles_to_remove]
        self.bugs = [b for i, b in enumerate(self.bugs) if i not in bugs_to_remove]

        # Bug-player collisions
        if self.player_invincibility_timer <= 0:
            for bug in self.bugs:
                if self.player_pos.distance_to(bug['pos']) < self.PLAYER_SIZE + self.BUG_SIZE:
                    # sound: player_hurt.wav
                    self.player_health -= 1
                    self.player_invincibility_timer = self.PLAYER_INVINCIBILITY_FRAMES
                    reward -= 10.0 # Penalty for getting hit
                    for _ in range(30): # Player hit effect
                        self._create_particle(self.player_pos, self.COLOR_PLAYER, 25, 4.0)
                    break # Only take damage from one bug per frame
        
        return reward

    def _update_difficulty(self):
        stage = self.bugs_killed_total // 5
        self.current_bug_speed = min(self.BUG_MAX_SPEED, self.BUG_INITIAL_SPEED + stage * self.BUG_SPEED_INCREMENT)
        self.current_spawn_interval = max(self.MIN_SPAWN_INTERVAL, self.INITIAL_SPAWN_INTERVAL - stage * self.SPAWN_INTERVAL_DECREMENT)

    def _check_termination(self):
        if self.game_over:
            return True
        if self.player_health <= 0:
            # sound: game_over.wav
            self.game_over = True
            self.game_won = False
            return True
        if self.bugs_killed_total >= self.WIN_CONDITION_KILLS:
            # sound: victory.wav
            self.game_over = True
            self.game_won = True
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            self.game_won = False
            return True
        return False
        
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_arena()
        self._render_particles()
        self._render_projectiles()
        self._render_bugs()
        self._render_player()
        self._render_ui()
        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.player_health,
            "bugs_killed": self.bugs_killed_total,
            "stage": self.bugs_killed_total // 5
        }

    def _render_arena(self):
        pygame.draw.rect(self.screen, self.COLOR_ARENA_BORDER, (self.ARENA_MARGIN, self.ARENA_MARGIN, self.WIDTH - 2 * self.ARENA_MARGIN, self.HEIGHT - 2 * self.ARENA_MARGIN), 2)

    def _render_player(self):
        # Flash when invincible
        if self.player_invincibility_timer > 0 and self.steps % 10 < 5:
            return

        pos = (int(self.player_pos.x), int(self.player_pos.y))
        
        # Draw main body with antialiasing
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.PLAYER_SIZE, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.PLAYER_SIZE, self.COLOR_PLAYER)

        # Draw gun barrel
        gun_end = self.player_pos + self.player_aim_direction * (self.PLAYER_SIZE + 4)
        pygame.draw.line(self.screen, self.COLOR_PLAYER_GUN, pos, (int(gun_end.x), int(gun_end.y)), 4)
        
    def _render_bugs(self):
        for bug in self.bugs:
            pos = bug['pos']
            p1 = pos + pygame.math.Vector2(0, -self.BUG_SIZE).rotate(-math.degrees(math.atan2(pos.y - self.player_pos.y, pos.x - self.player_pos.x)) - 90)
            p2 = pos + pygame.math.Vector2(-self.BUG_SIZE * 0.866, self.BUG_SIZE * 0.5).rotate(-math.degrees(math.atan2(pos.y - self.player_pos.y, pos.x - self.player_pos.x)) - 90)
            p3 = pos + pygame.math.Vector2(self.BUG_SIZE * 0.866, self.BUG_SIZE * 0.5).rotate(-math.degrees(math.atan2(pos.y - self.player_pos.y, pos.x - self.player_pos.x)) - 90)
            
            # Use antialiased triangles for smooth rotation
            pygame.gfxdraw.aatrigon(self.screen, int(p1.x), int(p1.y), int(p2.x), int(p2.y), int(p3.x), int(p3.y), self.COLOR_BUG)
            pygame.gfxdraw.filled_trigon(self.screen, int(p1.x), int(p1.y), int(p2.x), int(p2.y), int(p3.x), int(p3.y), self.COLOR_BUG)

    def _render_projectiles(self):
        for p in self.projectiles:
            pos = (int(p['pos'].x), int(p['pos'].y))
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.PROJECTILE_SIZE, self.COLOR_PROJECTILE)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.PROJECTILE_SIZE, self.COLOR_PROJECTILE)

    def _create_particle(self, pos, color, lifespan, max_speed):
        angle = self.np_random.uniform(0, 2 * math.pi)
        speed = self.np_random.uniform(0.5, max_speed)
        vel = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * speed
        self.particles.append({'pos': pos.copy(), 'vel': vel, 'lifespan': lifespan, 'color': color})

    def _render_particles(self):
        for p in self.particles:
            size = max(0, int(p['lifespan'] * 0.2))
            if size > 0:
                pygame.draw.circle(self.screen, p['color'], (int(p['pos'].x), int(p['pos'].y)), size)

    def _render_ui(self):
        # Health Bar
        health_bar_width = 100
        health_pct = self.player_health / self.PLAYER_MAX_HEALTH
        current_health_width = int(health_bar_width * health_pct)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (15, 15, health_bar_width, 20))
        if current_health_width > 0:
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (15, 15, current_health_width, 20))

        # Score
        score_surf = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_SCORE)
        self.screen.blit(score_surf, (self.WIDTH // 2 - score_surf.get_width() // 2, 15))

        # Bug Counter
        bug_surf = self.font_ui.render(f"KILLS: {self.bugs_killed_total}/{self.WIN_CONDITION_KILLS}", True, self.COLOR_TEXT)
        self.screen.blit(bug_surf, (self.WIDTH - bug_surf.get_width() - 15, 15))
        
    def _render_game_over(self):
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))

        text = "YOU WIN!" if self.game_won else "GAME OVER"
        color = self.COLOR_PLAYER if self.game_won else self.COLOR_BUG
        
        text_surf = self.font_game_over.render(text, True, color)
        text_rect = text_surf.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
        self.screen.blit(text_surf, text_rect)
    
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

if __name__ == '__main__':
    # This block allows you to play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Create a Pygame window to display the environment
    pygame.display.set_caption("Bug Hunt Arena")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    terminated = False
    
    # Game loop
    while not terminated:
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
        
        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)

        # --- Rendering ---
        # The observation is already a rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling (for closing the window) ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
    
    print(f"Game Over! Final Info: {info}")
    env.close()