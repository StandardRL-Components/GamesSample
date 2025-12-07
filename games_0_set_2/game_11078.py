import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T16:34:15.439779
# Source Brief: brief_01078.md
# Brief Index: 1078
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

# --- Helper Classes ---

class Territory:
    """Represents a single hexagonal territory on the board."""
    def __init__(self, q, r, owner, screen_pos, colors):
        self.q = q
        self.r = r
        self.owner = owner  # 0: Neutral, 1: Player, 2: Enemy
        self.screen_pos = screen_pos
        self.health = 100
        self.max_health = 100
        self.has_mech = False
        self.mech_anim_progress = 0.0
        self.colors = colors

        # Animation state
        self.current_color_tuple = self.colors['neutral']
        self.target_color_tuple = self.colors[self.get_owner_str()]
        self.color_lerp_progress = 1.0
        self.height_anim = 0.0 # For dynamic shrinking/growing effect

    def get_owner_str(self):
        return {0: 'neutral', 1: 'player', 2: 'enemy'}[self.owner]

    def set_owner(self, new_owner):
        if self.owner != new_owner:
            self.owner = new_owner
            self.target_color_tuple = self.colors[self.get_owner_str()]
            self.color_lerp_progress = 0.0
            self.has_mech = False # Mechs are destroyed on capture

    def damage(self, amount):
        self.health = max(0, self.health - amount)
        return self.health == 0

    def update(self):
        # Animate color transition
        if self.color_lerp_progress < 1.0:
            self.color_lerp_progress = min(1.0, self.color_lerp_progress + 0.05)
            self.current_color_tuple = tuple(
                int(a + (b - a) * self.color_lerp_progress)
                for a, b in zip(self.current_color_tuple, self.target_color_tuple)
            )
        # Animate mech deployment
        if self.has_mech and self.mech_anim_progress < 1.0:
            self.mech_anim_progress = min(1.0, self.mech_anim_progress + 0.1)
        elif not self.has_mech and self.mech_anim_progress > 0.0:
            self.mech_anim_progress = max(0.0, self.mech_anim_progress - 0.1)

class Projectile:
    """Represents a shot from one territory to another."""
    def __init__(self, start_pos, end_pos, color, speed, damage, owner):
        self.start_pos = np.array(start_pos, dtype=float)
        self.end_pos = np.array(end_pos, dtype=float)
        self.current_pos = self.start_pos.copy()
        self.color = color
        self.speed = speed
        self.damage = damage
        self.owner = owner # 1 for player, 2 for enemy
        self.total_dist = np.linalg.norm(self.end_pos - self.start_pos)
        self.direction = (self.end_pos - self.start_pos) / self.total_dist if self.total_dist > 0 else np.array([0,0])
        self.progress = 0.0

    def update(self):
        self.progress += self.speed
        self.current_pos = self.start_pos + self.direction * self.progress
        return np.linalg.norm(self.current_pos - self.start_pos) >= self.total_dist

class Particle:
    """A single particle for effects like explosions."""
    def __init__(self, pos, vel, color, lifetime):
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array(vel, dtype=float)
        self.color = color
        self.lifetime = lifetime
        self.max_lifetime = lifetime

    def update(self):
        self.pos += self.vel
        self.vel *= 0.95  # Damping
        self.lifetime -= 1
        return self.lifetime <= 0

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "A turn-based strategy game where you conquer hexagonal territories. "
        "Build mechs, research technology, and eliminate your opponent to claim victory."
    )
    user_guide = (
        "Use arrow keys (↑↓←→) to select a territory. "
        "Press Shift to cycle through actions and Space to confirm your choice."
    )
    auto_advance = False

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    HEX_SIZE = 22
    BOARD_COLS, BOARD_ROWS = 9, 6
    MAX_STEPS = 1000

    COLORS = {
        'bg': (10, 15, 30),
        'player': (0, 150, 255),
        'enemy': (255, 50, 50),
        'neutral': (70, 80, 100),
        'selected': (255, 255, 0),
        'player_side': (0, 100, 180),
        'enemy_side': (180, 30, 30),
        'neutral_side': (40, 50, 70),
        'grid': (30, 40, 60),
        'text': (220, 220, 240),
        'text_dark': (100, 100, 120),
        'time_dilated': (180, 0, 255),
    }

    ACTION_MODES = ['ATTACK', 'BUILD MECH', 'RESEARCH', 'END TURN']
    ACTION_COSTS = {'ATTACK': 10, 'BUILD MECH': 50, 'RESEARCH': 75}

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_s = pygame.font.Font(None, 20)
        self.font_m = pygame.font.Font(None, 28)
        self.font_l = pygame.font.Font(None, 48)

        self.board = {}
        self.selected_hex_coords = (0, 0)
        self.current_action_mode_idx = 0
        self.player_resources = 0
        self.player_tech_level = 1
        self.enemy_attack_chance_base = 0.25
        self.turn = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_status = 0 # 0: ongoing, 1: win, -1: loss
        self.pending_rewards = 0.0

        self.last_space_held = False
        self.last_shift_held = False
        
        self.particles = []
        self.projectiles = []
        self.nebula_stars = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_status = 0
        self.turn = 1
        
        self.player_resources = 100
        self.player_tech_level = 1
        
        self.last_space_held = False
        self.last_shift_held = False
        
        self.particles.clear()
        self.projectiles.clear()
        
        self._setup_board()
        self.selected_hex_coords = self._get_initial_player_hex()
        
        self.nebula_stars = [
            (self.np_random.integers(0, self.WIDTH), self.np_random.integers(0, self.HEIGHT), self.np_random.integers(1, 3))
            for _ in range(150)
        ]

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.pending_rewards = 0.0
        
        turn_ended = self._handle_player_input(action)

        if turn_ended:
            self.turn += 1
            self._run_enemy_turn()

        self._update_animations()
        
        reward = self.pending_rewards
        self.score += reward
        
        terminated = self._check_termination() or self.steps >= self.MAX_STEPS
        truncated = self.steps >= self.MAX_STEPS
        if terminated or truncated:
            self.game_over = True
            if self.win_status == 1:
                reward += 100
                self.score += 100
            elif self.win_status == -1:
                reward -= 100
                self.score -= 100

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _get_observation(self):
        self._render_background()
        self._render_board()
        for p in self.projectiles:
            self._render_projectile(p)
        for p in self.particles:
            self._render_particle(p)
        self._render_ui()
        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "turn": self.turn}
    
    # --- Game Logic ---
    
    def _setup_board(self):
        self.board.clear()
        start_x = self.WIDTH // 2 - (self.BOARD_COLS / 2) * self.HEX_SIZE * 1.75
        start_y = self.HEIGHT // 2 - (self.BOARD_ROWS / 2) * self.HEX_SIZE * math.sqrt(3) / 2
        
        for r in range(self.BOARD_ROWS):
            for q in range(self.BOARD_COLS):
                offset = r % 2 * self.HEX_SIZE * 0.866
                x = start_x + q * self.HEX_SIZE * 1.75 + offset
                y = start_y + r * self.HEX_SIZE * 1.5
                
                if (q, r) in self.board: continue

                owner = 0 # Neutral
                self.board[(q, r)] = Territory(q, r, owner, (x, y), self.COLORS)

        player_start = self._get_initial_player_hex()
        enemy_start = (self.BOARD_COLS - 1, self.BOARD_ROWS - 1 - player_start[1])
        
        self.board[player_start].set_owner(1)
        self.board[enemy_start].set_owner(2)
        
        for terr in self.board.values():
            terr.health = terr.max_health
            terr.has_mech = False
            terr.mech_anim_progress = 0.0
            terr.color_lerp_progress = 1.0
            terr.current_color_tuple = self.COLORS[terr.get_owner_str()]

    def _get_initial_player_hex(self):
        return (0, self.np_random.integers(1, self.BOARD_ROWS - 1))

    def _handle_player_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.last_space_held
        shift_pressed = shift_held and not self.last_shift_held
        
        self.last_space_held = space_held
        self.last_shift_held = shift_held

        if shift_pressed:
            self.current_action_mode_idx = (self.current_action_mode_idx + 1) % len(self.ACTION_MODES)
            return False

        if movement != 0:
            q, r = self.selected_hex_coords
            if movement == 1: r -= 1 # Up
            elif movement == 2: r += 1 # Down
            elif movement == 3: q -= 1 # Left
            elif movement == 4: q += 1 # Right
            
            if (q, r) in self.board:
                self.selected_hex_coords = (q, r)
            return False

        if space_pressed:
            return self._execute_player_action()

        return False

    def _execute_player_action(self):
        mode = self.ACTION_MODES[self.current_action_mode_idx]
        cost = self.ACTION_COSTS.get(mode, 0)

        if self.player_resources < cost:
            return False

        q, r = self.selected_hex_coords
        selected_territory = self.board.get((q, r))
        
        if mode == 'END TURN':
            self.player_resources += 10 + len(self._get_player_territories()) * 2 # Income
            return True

        if mode == 'ATTACK' and selected_territory and selected_territory.owner == 2:
            player_hexes = self._get_player_territories()
            if not player_hexes: return False
            
            attacker_hex = min(player_hexes, key=lambda h: np.linalg.norm(np.array(h.screen_pos) - np.array(selected_territory.screen_pos)))
            
            self.player_resources -= cost
            damage = 25 + self.player_tech_level * 10
            if attacker_hex.has_mech: damage *= 1.5

            self.projectiles.append(Projectile(attacker_hex.screen_pos, selected_territory.screen_pos, self.COLORS['player'], 15, damage, 1))
            return True

        if mode == 'BUILD MECH' and selected_territory and selected_territory.owner == 1 and not selected_territory.has_mech:
            self.player_resources -= cost
            selected_territory.has_mech = True
            return True

        if mode == 'RESEARCH' and self.player_tech_level < 5:
            self.player_resources -= cost
            self.player_tech_level += 1
            self.pending_rewards += 5
            return True

        return False

    def _run_enemy_turn(self):
        enemy_territories = self._get_enemy_territories()
        player_territories = self._get_player_territories()
        
        if not enemy_territories or not player_territories:
            return

        # Simple AI: attack a random player territory
        if self.np_random.random() < self.enemy_attack_chance_base + (self.turn / 200):
            attacker = self.np_random.choice(enemy_territories)
            target = min(player_territories, key=lambda t: t.health) # Prioritize weak targets
            
            damage = 20
            if attacker.has_mech: damage *= 1.5

            self.projectiles.append(Projectile(attacker.screen_pos, target.screen_pos, self.COLORS['enemy'], 15, damage, 2))
        
        # Enemy resource gain and build logic
        enemy_resources = 10 + len(enemy_territories) * 2
        if enemy_resources >= self.ACTION_COSTS['BUILD MECH'] and self.np_random.random() < 0.2:
            deploy_candidates = [t for t in enemy_territories if not t.has_mech]
            if deploy_candidates:
                self.np_random.choice(deploy_candidates).has_mech = True

    def _update_animations(self):
        # Update territories
        for terr in self.board.values():
            terr.update()

        # Update projectiles
        for proj in self.projectiles[:]:
            if proj.update():
                self.projectiles.remove(proj)
                target_hex = self._find_hex_at_pos(proj.end_pos)
                if target_hex:
                    is_fatal = target_hex.damage(proj.damage)
                    self._create_explosion(target_hex.screen_pos, proj.color)
                    
                    if proj.owner == 1: # Player shot
                        self.pending_rewards += 0.1
                        if is_fatal and target_hex.owner != 1:
                            target_hex.set_owner(1)
                            self.pending_rewards += 1
                    elif proj.owner == 2: # Enemy shot
                        if is_fatal and target_hex.owner != 2:
                            self.pending_rewards -= 0.1 # Penalty for losing a territory
                            target_hex.set_owner(2)
        
        # Update particles
        for part in self.particles[:]:
            if part.update():
                self.particles.remove(part)

    def _check_termination(self):
        player_hexes = self._get_player_territories()
        enemy_hexes = self._get_enemy_territories()
        if not player_hexes and self.turn > 1:
            self.win_status = -1
            return True
        if not enemy_hexes and self.turn > 1:
            self.win_status = 1
            return True
        return False

    # --- Helpers ---

    def _find_hex_at_pos(self, pos):
        for terr in self.board.values():
            if np.linalg.norm(np.array(pos) - np.array(terr.screen_pos)) < self.HEX_SIZE:
                return terr
        return None

    def _get_player_territories(self):
        return [t for t in self.board.values() if t.owner == 1]
    
    def _get_enemy_territories(self):
        return [t for t in self.board.values() if t.owner == 2]

    def _create_explosion(self, pos, color):
        for _ in range(30):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifetime = self.np_random.integers(20, 40)
            self.particles.append(Particle(pos, vel, color, lifetime))

    def _world_to_screen(self, q, r):
        # This logic is pre-calculated and stored in Territory.screen_pos
        return self.board[(q,r)].screen_pos

    # --- Rendering ---

    def _render_background(self):
        self.screen.fill(self.COLORS['bg'])
        for x, y, size in self.nebula_stars:
            c = self.np_random.integers(20, 40)
            pygame.draw.circle(self.screen, (c, c, c+10), (x, y), size)

    def _render_board(self):
        for (q, r), terr in sorted(self.board.items(), key=lambda item: item[0][1]):
            self._render_hex(terr)
        # Render selection cursor on top
        selected_terr = self.board.get(self.selected_hex_coords)
        if selected_terr:
            self._render_hex_outline(selected_terr.screen_pos, self.HEX_SIZE, self.COLORS['selected'], 3)
    
    def _render_hex(self, terr):
        x, y = terr.screen_pos
        size = self.HEX_SIZE
        
        top_color = terr.current_color_tuple
        side_color = self.COLORS[f"{terr.get_owner_str()}_side"]
        
        points = [(x + size * math.cos(math.pi / 180 * (60 * i + 30)), y + size * math.sin(math.pi / 180 * (60 * i + 30))) for i in range(6)]
        
        side_height = 8
        bottom_points = [(p[0], p[1] + side_height) for p in points]
        
        # Draw sides
        for i in range(6):
            p1 = points[i]
            p2 = points[(i + 1) % 6]
            bp1 = bottom_points[i]
            bp2 = bottom_points[(i + 1) % 6]
            # Only draw visible sides (1, 2, 3)
            if i in [1, 2, 3]:
                pygame.draw.polygon(self.screen, side_color, [p1, p2, bp2, bp1])

        # Draw top
        pygame.gfxdraw.filled_polygon(self.screen, points, top_color)
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLORS['grid'])
        
        # Draw health bar
        if terr.owner != 0:
            bar_width = size * 1.5
            bar_height = 4
            bar_x = x - bar_width / 2
            bar_y = y + size * 0.8
            health_ratio = terr.health / terr.max_health
            pygame.draw.rect(self.screen, (50, 50, 50), (bar_x, bar_y, bar_width, bar_height))
            pygame.draw.rect(self.screen, top_color, (bar_x, bar_y, bar_width * health_ratio, bar_height))
        
        # Draw mech
        if terr.mech_anim_progress > 0:
            self._render_mech(terr, terr.mech_anim_progress)

    def _render_mech(self, terr, progress):
        x, y = terr.screen_pos
        size = self.HEX_SIZE * 0.6 * progress
        color = self.COLORS[terr.get_owner_str()]
        dark_color = tuple(c*0.6 for c in color)
        y_offset = -10 * progress
        
        # Body
        pygame.draw.circle(self.screen, dark_color, (x, y + y_offset), size)
        pygame.draw.circle(self.screen, color, (x, y + y_offset), size * 0.8)
        # Glow
        s = int(size * 1.5)
        if s > 0:
            glow_surf = pygame.Surface((s*2, s*2), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, (*color, 50), (s, s), s)
            self.screen.blit(glow_surf, (x - s, y + y_offset - s))

    def _render_hex_outline(self, pos, size, color, width):
        x, y = pos
        points = [(x + size * math.cos(math.pi / 180 * (60 * i + 30)), y + size * math.sin(math.pi / 180 * (60 * i + 30))) for i in range(6)]
        pygame.draw.lines(self.screen, color, True, points, width)

    def _render_projectile(self, proj):
        start = proj.current_pos - proj.direction * 10
        end = proj.current_pos
        pygame.draw.line(self.screen, proj.color, start, end, 3)
        # Glow
        s = pygame.Surface((24, 24), pygame.SRCALPHA)
        pygame.draw.circle(s, (*proj.color, 100), (12, 12), 8)
        pygame.draw.circle(s, (*proj.color, 50), (12, 12), 12)
        self.screen.blit(s, (proj.current_pos[0]-12, proj.current_pos[1]-12))

    def _render_particle(self, part):
        alpha = int(255 * (part.lifetime / part.max_lifetime))
        color = (*part.color, alpha)
        size = int(5 * (part.lifetime / part.max_lifetime))
        if size > 0:
            temp_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (size, size), size)
            self.screen.blit(temp_surf, (part.pos[0] - size, part.pos[1] - size))

    def _render_ui(self):
        # Top-left: Resources & Score
        res_text = self.font_m.render(f"RES: {self.player_resources}", True, self.COLORS['text'])
        score_text = self.font_s.render(f"SCORE: {int(self.score)}", True, self.COLORS['text'])
        self.screen.blit(res_text, (10, 10))
        self.screen.blit(score_text, (10, 35))

        # Top-right: Turn & Tech
        turn_text = self.font_m.render(f"TURN: {self.turn}", True, self.COLORS['text'])
        tech_text = self.font_s.render(f"TECH: LVL {self.player_tech_level}", True, self.COLORS['text'])
        self.screen.blit(turn_text, (self.WIDTH - turn_text.get_width() - 10, 10))
        self.screen.blit(tech_text, (self.WIDTH - tech_text.get_width() - 10, 35))

        # Bottom-center: Action Mode & Info
        mode_text = self.ACTION_MODES[self.current_action_mode_idx]
        cost = self.ACTION_COSTS.get(mode_text, 0)
        cost_str = f" (Cost: {cost})" if cost > 0 else ""
        
        action_surf = self.font_m.render(f"ACTION: {mode_text}{cost_str}", True, self.COLORS['text'])
        action_rect = action_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT - 50))
        self.screen.blit(action_surf, action_rect)

        help_text = self.font_s.render("ARROWS: Select | SHIFT: Cycle Action | SPACE: Confirm", True, self.COLORS['text_dark'])
        help_rect = help_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT - 20))
        self.screen.blit(help_text, help_rect)

    def _render_game_over(self):
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        
        status_text = "VICTORY" if self.win_status == 1 else "DEFEAT"
        color = self.COLORS['player'] if self.win_status == 1 else self.COLORS['enemy']
        
        status_surf = self.font_l.render(status_text, True, color)
        status_rect = status_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2 - 20))
        self.screen.blit(status_surf, status_rect)
        
        score_surf = self.font_m.render(f"Final Score: {int(self.score)}", True, self.COLORS['text'])
        score_rect = score_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2 + 30))
        self.screen.blit(score_surf, score_rect)

if __name__ == '__main__':
    # --- Example Usage ---
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play ---
    # This requires a display, so we unset the dummy driver
    os.environ.pop("SDL_VIDEODRIVER", None)
    pygame.quit() # Quit the dummy instance
    pygame.init() # Re-init with display
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Tactical Conquest")
    clock = pygame.time.Clock()
    
    obs, info = env.reset()
    done = False
    
    # Re-init fonts for the new pygame instance
    env.font_s = pygame.font.Font(None, 20)
    env.font_m = pygame.font.Font(None, 28)
    env.font_l = pygame.font.Font(None, 48)

    while not done:
        action = [0, 0, 0] # Default no-op
        action_taken = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                action_taken = True
                if event.key == pygame.K_UP:
                    action[0] = 1
                elif event.key == pygame.K_DOWN:
                    action[0] = 2
                elif event.key == pygame.K_LEFT:
                    action[0] = 3
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4
                elif event.key == pygame.K_SPACE:
                    action[1] = 1
                elif event.key in [pygame.K_LSHIFT, pygame.K_RSHIFT]:
                    action[2] = 1

        if action_taken:
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                print(f"Game Over! Final Info: {info}")
                # Keep rendering the final screen for a bit
                final_obs = env._get_observation()
                surf = pygame.surfarray.make_surface(np.transpose(final_obs, (1, 0, 2)))
                screen.blit(surf, (0, 0))
                pygame.display.flip()
                pygame.time.wait(3000)
                done = True
        else:
            # Since auto_advance is False, we only need to render, not step
            obs = env._get_observation()

        if not done:
            # Render the observation from the environment
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
        
        clock.tick(30) # Run at 30 FPS

    pygame.quit()