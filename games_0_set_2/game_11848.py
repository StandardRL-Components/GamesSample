import gymnasium as gym
import os
import pygame
import numpy as np
import math
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
pygame.init()
pygame.font.init()

from gymnasium.spaces import MultiDiscrete
import pygame.gfxdraw


class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player controls a powerful shotgun to
    blast apart a ragdolling centipede in a destructible environment.

    The core gameplay loop involves precise aiming, managing recoil, and
    strategically destroying both the centipede and the terrain to create
    clear lines of sight. The game prioritizes satisfying physics and
    visual feedback.
    """
    game_description = (
        "Control a powerful shotgun to blast apart a ragdolling centipede in a destructible environment."
    )
    user_guide = (
        "Controls: Use ↑/↓ arrow keys to aim. Press space to shoot."
    )
    auto_advance = True
    metadata = {"render_modes": ["rgb_array"]}

    # --- CONSTANTS ---
    # Colors
    COLOR_BG = (15, 10, 25)
    COLOR_TERRAIN = (70, 50, 40)
    COLOR_TERRAIN_DMG = (50, 35, 25)
    COLOR_PLAYER = (230, 230, 255)
    COLOR_PLAYER_GLOW = (100, 100, 220)
    COLOR_CENTIPEDE = (100, 255, 100)
    COLOR_CENTIPEDE_GLOW = (50, 150, 50)
    COLOR_BULLET = (255, 255, 100)
    COLOR_BULLET_GLOW = (200, 200, 0)
    COLOR_MUZZLE_FLASH = (255, 200, 150)
    COLOR_HEART = (255, 50, 50)
    COLOR_UI_TEXT = (220, 220, 220)

    # Screen Dimensions
    WIDTH, HEIGHT = 640, 400

    # Game Parameters
    MAX_STEPS = 1000
    PLAYER_POS = pygame.math.Vector2(60, HEIGHT / 2)
    PLAYER_HEALTH_START = 3
    AIM_SPEED = 0.05  # Radians per step
    RECOIL_AMOUNT = 0.25
    RECOIL_DECAY = 0.85
    SHOTGUN_SPREAD = 0.2  # Radians
    SHOTGUN_PELLETS = 7
    BULLET_SPEED = 25
    CENTIPEDE_LENGTH = 10
    CENTIPEDE_SEGMENT_RADIUS = 12
    CENTIPEDE_START_SPEED = 1.0
    CENTIPEDE_SPEED_INCREASE_INTERVAL = 200
    CENTIPEDE_SPEED_INCREASE_AMOUNT = 0.1
    GRAVITY = 0.4
    FRICTION = 0.98
    TERRAIN_BLOCK_SIZE = 20
    TERRAIN_COLS = 16
    TERRAIN_ROWS = 20
    TERRAIN_START_COL = 10

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame Setup
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 48, bold=True)

        # State variables are initialized in reset()
        self.player_angle = 0
        self.player_health = 0
        self.recoil = 0
        self.last_space_held = False
        self.muzzle_flash_timer = 0
        self.screen_shake_timer = 0
        self.centipede_segments = []
        self.bullets = []
        self.particles = []
        self.terrain = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.centipede_speed = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False

        self.player_angle = 0
        self.player_health = self.PLAYER_HEALTH_START
        self.recoil = 0
        self.last_space_held = False
        self.muzzle_flash_timer = 0
        self.screen_shake_timer = 0

        self.bullets.clear()
        self.particles.clear()
        
        self.centipede_speed = self.CENTIPEDE_START_SPEED
        self._spawn_centipede()
        self._spawn_terrain()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            # If the game is over, no actions should have any effect.
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0

        # --- Handle Input ---
        self._handle_input(action)

        # --- Update Game State ---
        self._update_physics()
        reward += self._check_collisions()
        
        # --- Difficulty Scaling ---
        if self.steps > 0 and self.steps % self.CENTIPEDE_SPEED_INCREASE_INTERVAL == 0:
            self.centipede_speed += self.CENTIPEDE_SPEED_INCREASE_AMOUNT

        # --- Termination Check ---
        terminated = self._check_termination()
        if terminated:
            if self.player_health <= 0:
                reward -= 100  # Penalty for losing
            elif not any(s['attached'] for s in self.centipede_segments):
                reward += 100  # Bonus for winning
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1

        # Aiming
        if movement == 1:  # Up
            self.player_angle -= self.AIM_SPEED
        elif movement == 2:  # Down
            self.player_angle += self.AIM_SPEED
        self.player_angle = max(-math.pi / 2, min(math.pi / 2, self.player_angle))

        # Shooting (on button press, not hold)
        if space_held and not self.last_space_held:
            self._fire_shotgun()
        self.last_space_held = space_held

    def _fire_shotgun(self):
        # sfx: SHOTGUN_BLAST.WAV
        self.recoil = self.RECOIL_AMOUNT
        self.muzzle_flash_timer = 2
        self.screen_shake_timer = 4
        
        for _ in range(self.SHOTGUN_PELLETS):
            angle_offset = self.np_random.uniform(-self.SHOTGUN_SPREAD, self.SHOTGUN_SPREAD)
            bullet_angle = self.player_angle + angle_offset
            vel = pygame.math.Vector2(math.cos(bullet_angle), math.sin(bullet_angle)) * self.BULLET_SPEED
            self.bullets.append({'pos': self.PLAYER_POS.copy(), 'vel': vel})
        
        # Base reward for shooting, will be counteracted if it's a miss
        return -0.01

    def _update_physics(self):
        # Update Recoil
        self.recoil *= self.RECOIL_DECAY
        if self.muzzle_flash_timer > 0: self.muzzle_flash_timer -= 1
        if self.screen_shake_timer > 0: self.screen_shake_timer -= 1

        # Update Bullets
        self.bullets = [b for b in self.bullets if 0 < b['pos'].x < self.WIDTH and 0 < b['pos'].y < self.HEIGHT]
        for b in self.bullets:
            b['pos'] += b['vel']

        # Update Particles
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'] *= self.FRICTION
            p['life'] -= 1
            p['size'] = max(0, p['size'] - 0.1)

        # Update Centipede
        self._update_centipede_movement()

    def _update_centipede_movement(self):
        target_pos = self.PLAYER_POS
        for i, seg in enumerate(self.centipede_segments):
            if seg['attached']:
                if i == 0: # Head segment
                    direction = (target_pos - seg['pos']).normalize()
                    seg['pos'] += direction * self.centipede_speed
                else: # Body segments
                    prev_seg = self.centipede_segments[i-1]
                    dist_vec = prev_seg['pos'] - seg['pos']
                    if dist_vec.length() > self.CENTIPEDE_SEGMENT_RADIUS * 1.5:
                        direction = dist_vec.normalize()
                        seg['pos'] += direction * self.centipede_speed * 1.1 # Catch up
            else: # Ragdoll physics for detached segments
                seg['vel'].y += self.GRAVITY
                seg['pos'] += seg['vel']
                seg['vel'] *= self.FRICTION
                # Ground collision
                if seg['pos'].y > self.HEIGHT - self.CENTIPEDE_SEGMENT_RADIUS:
                    seg['pos'].y = self.HEIGHT - self.CENTIPEDE_SEGMENT_RADIUS
                    seg['vel'].y *= -0.5
                # Wall collisions
                if seg['pos'].x < self.CENTIPEDE_SEGMENT_RADIUS or seg['pos'].x > self.WIDTH - self.CENTIPEDE_SEGMENT_RADIUS:
                    seg['vel'].x *= -0.5

    def _check_collisions(self):
        reward = 0
        bullets_to_remove = []

        # Bullet collisions
        for i, bullet in enumerate(self.bullets):
            if i in bullets_to_remove: continue

            # Bullet vs Terrain
            grid_x = int(bullet['pos'].x / self.TERRAIN_BLOCK_SIZE)
            grid_y = int(bullet['pos'].y / self.TERRAIN_BLOCK_SIZE)
            if 0 <= grid_y < self.TERRAIN_ROWS and 0 <= grid_x < self.TERRAIN_COLS and self.terrain[grid_y][grid_x]:
                self.terrain[grid_y][grid_x] = 0
                bullets_to_remove.append(i)
                self._spawn_particles(bullet['pos'], 10, (120, 80, 60), 2, 4)
                # sfx: TERRAIN_DESTROY.WAV
                continue

            # Bullet vs Centipede
            for j, seg in enumerate(self.centipede_segments):
                if seg['health'] > 0 and (bullet['pos'] - seg['pos']).length() < self.CENTIPEDE_SEGMENT_RADIUS:
                    bullets_to_remove.append(i)
                    seg['health'] -= 1
                    reward += 0.1  # Reward for hitting
                    self._spawn_particles(bullet['pos'], 15, self.COLOR_CENTIPEDE, 3, 5)
                    # sfx: FLESH_HIT.WAV
                    
                    if seg['health'] <= 0:
                        reward += 1.0  # Reward for destroying a segment
                        self.score += 10
                        self._detach_centipede(j, bullet['vel'])
                    break
        
        # Remove bullets that hit something
        self.bullets = [b for i, b in enumerate(self.bullets) if i not in bullets_to_remove]

        # Centipede vs Player
        for seg in self.centipede_segments:
            if seg['attached'] and (seg['pos'] - self.PLAYER_POS).length() < self.CENTIPEDE_SEGMENT_RADIUS + 10:
                self.player_health -= 1
                self.score -= 50
                reward -= 1.0  # Penalty for getting hit
                self.screen_shake_timer = 10
                # sfx: PLAYER_HURT.WAV
                self._detach_centipede(self.centipede_segments.index(seg), (seg['pos'] - self.PLAYER_POS).normalize() * 5)
                break
        
        return reward
    
    def _detach_centipede(self, index, impact_vel):
        if not self.centipede_segments[index]['attached']:
            return # Already detached
        
        # sfx: CENTIPEDE_SPLIT.WAV
        explosion_force = 5
        for i in range(index, len(self.centipede_segments)):
            seg = self.centipede_segments[i]
            if seg['attached']:
                seg['attached'] = False
                seg['health'] = 0
                # Apply physics impulse
                direction_from_impact = (seg['pos'] - self.centipede_segments[index]['pos']).normalize() if i != index else impact_vel.normalize()
                if direction_from_impact.length() == 0:
                    direction_from_impact = pygame.math.Vector2(self.np_random.uniform(-1, 1), self.np_random.uniform(-1, 1)).normalize()
                
                force = direction_from_impact * explosion_force * self.np_random.uniform(0.8, 1.2)
                seg['vel'] = force + impact_vel * 0.2
                self._spawn_particles(seg['pos'], 20, (255,100,50), 4, 6)

    def _check_termination(self):
        win = not any(s['attached'] for s in self.centipede_segments)
        lose = self.player_health <= 0
        timeout = self.steps >= self.MAX_STEPS
        return win or lose or timeout

    def _get_observation(self):
        render_offset = pygame.math.Vector2(0, 0)
        if self.screen_shake_timer > 0:
            render_offset.x = self.np_random.uniform(-5, 5)
            render_offset.y = self.np_random.uniform(-5, 5)

        # Clear screen
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_terrain(render_offset)
        self._render_particles(render_offset)
        self._render_centipede(render_offset)
        self._render_bullets(render_offset)
        self._render_player(render_offset)
        
        # Render UI (not affected by screen shake)
        self._render_ui()
        
        if self.game_over:
            self._render_game_over()

        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.player_health,
            "segments_left": sum(1 for s in self.centipede_segments if s['attached'])
        }

    def _spawn_centipede(self):
        self.centipede_segments.clear()
        start_x = self.WIDTH + 50
        start_y = self.HEIGHT / 2
        for i in range(self.CENTIPEDE_LENGTH):
            self.centipede_segments.append({
                'pos': pygame.math.Vector2(start_x + i * self.CENTIPEDE_SEGMENT_RADIUS * 1.5, start_y),
                'vel': pygame.math.Vector2(0, 0),
                'attached': True,
                'health': 2
            })
    
    def _spawn_terrain(self):
        self.terrain = [[0] * self.TERRAIN_COLS for _ in range(self.TERRAIN_ROWS)]
        for r in range(self.TERRAIN_ROWS):
            for c in range(self.TERRAIN_START_COL, self.TERRAIN_COLS):
                if self.np_random.random() > 0.3:
                    self.terrain[r][c] = 1

    def _spawn_particles(self, pos, count, color, min_speed, max_speed):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(min_speed, max_speed)
            vel = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'life': self.np_random.integers(15, 30),
                'size': self.np_random.uniform(3, 7),
                'color': color
            })

    # --- RENDER METHODS ---

    def _render_terrain(self, offset):
        for r in range(self.TERRAIN_ROWS):
            for c in range(self.TERRAIN_COLS):
                if self.terrain[r][c]:
                    rect = pygame.Rect(
                        offset.x + c * self.TERRAIN_BLOCK_SIZE,
                        offset.y + r * self.TERRAIN_BLOCK_SIZE,
                        self.TERRAIN_BLOCK_SIZE,
                        self.TERRAIN_BLOCK_SIZE
                    )
                    pygame.draw.rect(self.screen, self.COLOR_TERRAIN, rect)
                    pygame.draw.rect(self.screen, self.COLOR_TERRAIN_DMG, rect, 2)
    
    def _render_particles(self, offset):
        for p in self.particles:
            alpha = int(255 * (p['life'] / 30.0))
            color = (*p['color'], alpha)
            pos = (int(p['pos'].x + offset.x), int(p['pos'].y + offset.y))
            pygame.gfxdraw.filled_circle(self.screen, *pos, int(p['size']), color)

    def _render_centipede(self, offset):
        for seg in self.centipede_segments:
            pos = (int(seg['pos'].x + offset.x), int(seg['pos'].y + offset.y))
            radius = self.CENTIPEDE_SEGMENT_RADIUS
            
            # Glow effect
            glow_color = (*self.COLOR_CENTIPEDE_GLOW, 50)
            pygame.gfxdraw.filled_circle(self.screen, *pos, int(radius * 1.5), glow_color)
            
            # Main body
            body_color = self.COLOR_CENTIPEDE
            if seg['health'] == 1: # Damaged state
                body_color = (180, 255, 180) # Lighter green
            pygame.gfxdraw.filled_circle(self.screen, *pos, radius, body_color)
            pygame.gfxdraw.aacircle(self.screen, *pos, radius, body_color)

    def _render_bullets(self, offset):
        for b in self.bullets:
            pos = (int(b['pos'].x + offset.x), int(b['pos'].y + offset.y))
            
            # Glow
            glow_color = (*self.COLOR_BULLET_GLOW, 100)
            pygame.gfxdraw.filled_circle(self.screen, *pos, 5, glow_color)
            
            # Core
            pygame.gfxdraw.filled_circle(self.screen, *pos, 3, self.COLOR_BULLET)
            pygame.gfxdraw.aacircle(self.screen, *pos, 3, self.COLOR_BULLET)

    def _render_player(self, offset):
        gun_length = 40
        gun_width = 10
        
        # Recoil affects the base position
        recoil_offset = pygame.math.Vector2(math.cos(self.player_angle), math.sin(self.player_angle)) * -self.recoil * 20
        gun_base_pos = self.PLAYER_POS + offset + recoil_offset

        # Calculate gun polygon points
        angle = self.player_angle
        dx, dy = math.cos(angle), math.sin(angle)
        p1 = (gun_base_pos.x - dy * gun_width/2, gun_base_pos.y + dx * gun_width/2)
        p2 = (gun_base_pos.x + dy * gun_width/2, gun_base_pos.y - dx * gun_width/2)
        p3 = (p2[0] + dx * gun_length, p2[1] + dy * gun_length)
        p4 = (p1[0] + dx * gun_length, p1[1] + dy * gun_length)
        
        # Glow
        pygame.draw.circle(self.screen, self.COLOR_PLAYER_GLOW, (int(self.PLAYER_POS.x + offset.x), int(self.PLAYER_POS.y + offset.y)), 20, 0)
        
        # Gun
        pygame.gfxdraw.aapolygon(self.screen, [p1, p2, p3, p4], self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, [p1, p2, p3, p4], self.COLOR_PLAYER)
        
        # Base
        pygame.draw.circle(self.screen, self.COLOR_PLAYER, (int(self.PLAYER_POS.x + offset.x), int(self.PLAYER_POS.y + offset.y)), 12, 0)

        # Muzzle Flash
        if self.muzzle_flash_timer > 0:
            flash_pos = (p3[0] + p4[0]) / 2, (p3[1] + p4[1]) / 2
            size = self.np_random.integers(20, 30)
            points = []
            for i in range(8):
                rad = size if i % 2 == 0 else size / 2
                a = angle + i * math.pi / 4
                points.append((flash_pos[0] + rad * math.cos(a), flash_pos[1] + rad * math.sin(a)))
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_MUZZLE_FLASH)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Health
        for i in range(self.player_health):
            heart_pos = (self.WIDTH - 30 - i * 35, 15)
            pygame.gfxdraw.filled_polygon(self.screen, self._get_heart_points(heart_pos, 25), self.COLOR_HEART)
            pygame.gfxdraw.aapolygon(self.screen, self._get_heart_points(heart_pos, 25), self.COLOR_HEART)

    def _render_game_over(self):
        s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        s.fill((0,0,0,128))
        self.screen.blit(s, (0,0))
        
        win = not any(seg['attached'] for seg in self.centipede_segments)
        text = "VICTORY" if win else "GAME OVER"
        color = (100, 255, 100) if win else (255, 100, 100)
        
        game_over_text = self.font_game_over.render(text, True, color)
        text_rect = game_over_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
        self.screen.blit(game_over_text, text_rect)

    def _get_heart_points(self, pos, size):
        x, y = pos
        s = size
        return [
            (x, y + s*0.25), (x - s*0.5, y - s*0.25), (x - s*0.25, y - s*0.6),
            (x, y - s*0.25), (x + s*0.25, y - s*0.6), (x + s*0.5, y - s*0.25)
        ]

    def close(self):
        pygame.quit()


if __name__ == "__main__":
    # This block allows for interactive testing of the environment.
    # Controls:
    #   W/S or Up/Down Arrow: Aim
    #   Space: Shoot
    #   R: Reset
    #   Q: Quit
    
    # Set the video driver to a real one for interactive mode
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    pygame.display.set_caption("Centipede Annihilation")
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    clock = pygame.time.Clock()
    
    action = np.array([0, 0, 0]) # [movement, space, shift]
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    done = False
                if event.key == pygame.K_SPACE:
                    action[1] = 1
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_SPACE:
                    action[1] = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            action[0] = 1 # Up
        elif keys[pygame.K_s] or keys[pygame.K_DOWN]:
            action[0] = 2 # Down
        else:
            action[0] = 0 # No movement

        if not done:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        # Display the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Run at 30 FPS

    env.close()